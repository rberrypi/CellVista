#include "stdafx.h"
#include <QMessageBox>
#include "device_factory.h"
#include "slm_device.h"
#include <QDirIterator>
#include "asi_devices.h"
#include <unordered_set>
#include <QAbstractButton>
#include "compact_light_path.h"
bool acquisition::prompt_if_failure(const QString& message)
{
	if (!message.isEmpty())
	{
		QMessageBox msg_box;
		msg_box.setText(message);
		msg_box.setInformativeText("Continue?");
		msg_box.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		msg_box.setDefaultButton(QMessageBox::No);
		return msg_box.exec() == QMessageBox::Yes;
	}
	return true;
};

acquisition::preflight_info acquisition::preflight_checks(const used_channels& channels_used) const
{
	//preflight_checks
	assert_valid();
	const size_t resume = 0;
	std::vector<preflight_function> preflight_checks;
	//
	//Overwrite	
	const auto overwrite_experiment = [&]()
	{
		const auto dir = QString::fromStdString(output_dir);
		QStringList name_filter;
		name_filter << "*.tif";
		const auto not_empty = !QDir(dir).entryInfoList(name_filter).isEmpty();
		return not_empty ? QString("Files exist in %1").arg(dir) : QString();
	};
	preflight_checks.emplace_back(overwrite_experiment);
	//Has Items
	const auto has_items = [&]
	{
		return number_of_events() > 0 ? QString() : QString("Empty Acquisition");
	};
	preflight_checks.emplace_back(has_items);
	//Missing Channels 
	const auto has_missing_channels = [&]()
	{
		for (const auto& channel_idx : channels_used) {
			if (channel_idx >= ch.size())		//channel must be added to light path on screen
			{
				return QString("Channel %1 is Missing").arg(channel_idx);
			}
		}
		return QString();
	};
	preflight_checks.emplace_back(has_missing_channels);
	//Unused Channels
	const auto has_unused_channels = [&]()
	{
		// is this correct?
		const auto& channels_size = ch.size();
		if (channels_used.size() < channels_size)
		{
			QString message = "Unused Channels: ";
			std::vector<bool> missing_channels(channels_size, false);
			for (const auto& ch : channels_used)
			{
				missing_channels.at(ch) = true;
			}
			for (auto i = 0; i < missing_channels.size(); ++i) {
				if (!missing_channels.at(i))
				{
					message.append(QString::number(i)).append(", ");
				}
			}
			message.chop(2);
			return message;
		}
		return QString();
	};
	preflight_checks.emplace_back(has_unused_channels);
	//Repeated Settings (almost always an error)
	const auto has_repeated_channel_modes = [&]()
	{
		std::unordered_map<int, int> map_camera;
		std::unordered_map<phase_retrieval, phase_processing> map;
		auto failed = false;
		for (const auto& light_path : acquisition::ch)
		{
			if (light_path.retrieval == phase_retrieval::camera) {
				const auto scope_id = light_path.scope_channel;
				const auto path_id = light_path.light_path;
				if (map_camera.count(scope_id))
				{
					failed = true;
					break;
				}
				map_camera.insert({ scope_id, path_id });
			}
			else
			{
				const auto retrieval = light_path.retrieval;
				const auto processing = light_path.processing;
				if (map.count(retrieval))
				{
					if (map[retrieval] == processing)
					{
						failed = true;
						break;
					}
				}
				map.insert({ retrieval, processing });
			}
		}
		return failed ? "Two channels with a similar setup in the channels panel" : QString();
	};
	preflight_checks.emplace_back(has_repeated_channel_modes);
	//All points inside XY scanning range
	const auto all_inside_xy = [&]
	{
		const auto& points = cap;
		const auto predicate = [&](const capture_item& item) {return !D->scope->xy_drive->point_inside_range(item); };
		const auto bad_point = std::find_if(points.begin(), points.end(), predicate);
		const auto failed = bad_point != points.end();
		const auto outside_xy_range = QString("ROI %1 xy position is outside of xy range").arg(failed ? bad_point->roi : 0);
		return failed ? outside_xy_range : QString();//problems with DPM
	};
	preflight_checks.emplace_back(all_inside_xy);
	//All points inside Z scanning range	
	const auto all_inside_z = [&]
	{
		const auto& points = cap;
		const auto predicate = [&](const capture_item& item) {return !D->scope->z_drive->point_inside_range(item.z); };
		const auto bad_point = std::find_if(points.begin(), points.end(), predicate);
		const auto failed = bad_point != points.end();
		const auto outside_y_range = QString("ROI %1 z position is outside of z range").arg(failed ? bad_point->roi : 0);
		return failed ? outside_y_range : QString();//problems with DPM
	};
	preflight_checks.emplace_back(all_inside_z);
	//Has a phase mode
	const auto selection_has_phase = [&]()
	{
		auto bad = true;
		for (auto idx : channels_used)
		{
			const auto not_camera = ch.at(idx).retrieval != phase_retrieval::camera;
			if (not_camera)
			{
				bad = false;
				break;
			}
		}
		return bad ? "Not performing phase-shifting?" : QString();
	};
	preflight_checks.emplace_back(selection_has_phase);
	//Has a phase mode
	const auto filtering_raw_ft = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			bad = channel.is_raw_frame() && channel.do_band_pass;
			if (bad)
			{
				break;
			}
		}
		return bad ? "Fourier filtering on raw camera data?" : QString();
	};
	preflight_checks.emplace_back(filtering_raw_ft);
	//Unstable modulators
	const auto slm_stability_valid = [&]
	{
		auto bad = false;
		const auto vendor_time = D->max_vendor_stability_time();
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			if (channel.modulates_slm())
			{
				const auto& exposures = channel.exposures_and_delays;
				for (const auto& exposure : exposures)
				{
					bad = exposure.slm_stability < vendor_time;
					if (bad)
					{
						goto escape;
					}
				}
			}
		}
	escape:
		const auto prompt = QString("Using less than recommended SLM stability time of %1 ms?").arg(to_mili(vendor_time));
		return  bad ? prompt : QString();//problems with DPM
	};
	preflight_checks.emplace_back(slm_stability_valid);
	//Exposures are valid
	const auto exposures_valid = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			{
				const auto& exposures = channel.exposures_and_delays;
				for (const auto& exposure : exposures)
				{
					bad = exposure.exposure_time <= ms_to_chrono(0);
					if (bad)
					{
						goto escape;
					}
				}
			}
		}
	escape:
		return bad ? QString("Some frames have zero exposure?") : QString();
	};
	preflight_checks.emplace_back(exposures_valid);
	//Exposures are valid
	const auto modulation_stability_valid = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			if (channel.modulates_slm())
			{
				const auto& exposures = channel.exposures_and_delays;
				for (const auto& exposure : exposures)
				{
					bad = exposure.exposure_time <= ms_to_chrono(0);
					if (bad)
					{
						goto escape;
					}
				}
			}
		}
	escape:
		return bad ? QString("Some frames have zero modulator delay?") : QString();
	};
	preflight_checks.emplace_back(modulation_stability_valid);
	//
	const auto accidentally_using_off_channel = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			bad = channel.scope_channel == scope_channel_drive_settings::off_channel_idx;
			if (bad)
			{
				goto escape;
			}
		}
	escape:
		const auto prompt = QString("Warning, one of the channels will turn off the light?");
		return bad ? prompt : QString();
	};
	preflight_checks.emplace_back(accidentally_using_off_channel);
	//
	const auto all_dpm_valid = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			const auto is_dpm = phase_processing_setting::settings.at(channel.processing).is_a_dpm;
			if (is_dpm)
			{
				bad = !channel.dpm_phase_is_complete();
			}
			if (bad)
			{
				goto  escape;
			}
		}
	escape:
		const auto prompt = QString("Looks like a DPM channel has invalid DPM settings (this will probably result in a crash)");
		return bad ? prompt : QString();
	};
	preflight_checks.emplace_back(all_dpm_valid);
	//Normally we don't want to write the FT
	const auto check_for_ft = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			bad = channel.do_ft;
			if (bad)
			{
				goto  escape;
			}
		}
	escape:
		const auto prompt = QString("Warning, writing a spectrum instead of an image, this is almost never desired.");
		return bad ? prompt : QString();
	};
	preflight_checks.emplace_back(check_for_ft);
	//Don't acquire the reference ring for darkfield illumination
	const auto darkfield_mode_valid = [&]
	{
		auto bad = false;
		for (auto idx : channels_used)
		{
			const auto& channel = ch.at(idx);
			for (const auto& modulator : channel.modulator_settings)
			{
				bad = modulator.darkfield_display_mode!= darkfield_pattern_settings::darkfield_display_align_mode::darkfield;
				if (bad)
				{
					goto  escape;
				}
			}

		}
	escape:
		const auto prompt = QString("Warning, darkfield configured for alignment");
		return bad ? prompt : QString();
	};
	preflight_checks.emplace_back(darkfield_mode_valid);
	//
	const auto modulator_voltage_is_valid = [&]
	{
		const auto retarders = D->has_retarders();
		auto bad = false;
		{
			for (auto idx : channels_used)
			{
				const auto& settings = ch.at(idx);
				for (auto modulator_idx = 0; modulator_idx < settings.modulator_settings.size(); ++modulator_idx)
				{
					const auto valid_voltage = settings.modulator_settings.at(modulator_idx).valid_voltage();
					if (!valid_voltage  && retarders.at(modulator_idx))
					{
						bad = true;
						goto escape;
					}
				}
			}
		}
	escape:
		const auto prompt = QString("LCVR modulator voltage is invalid, set it before continuing?");
		return bad ? prompt : QString();
	};
	preflight_checks.emplace_back(modulator_voltage_is_valid);
	const auto predicate = [&](const preflight_function& test)
	{
		return prompt_if_failure(test());
	};
	const auto preflight_pass = std::all_of(preflight_checks.begin(), preflight_checks.end(), predicate);
	const preflight_info info = { preflight_pass,resume };
	return 	info;
}
