#include "stdafx.h"
#include "full_interface_gui.h"
#include "device_factory.h"
#include "camera_device.h"
#include "snake_iterator.h"
#include "capture_modes.h"
#include <QDirIterator>

#include "qli_runtime_error.h"
#include "ui_full_interface_gui.h"


const full_interface_gui::roi_switching_map full_interface_gui::roi_switching_settings =
{
	//should be icons, look at how its communicated in NI elements? (use fancy icons)
	{ channel_switching_order::switch_channel_per_roi, "Per ROI (RCXYZ) Fastest"},
	{ channel_switching_order::switch_channel_per_tile, "Per Tile (RXYZC) Slowest"},
	{ channel_switching_order::switch_channel_per_grid, "Per Grid XY (RZCXY)"},
	{ channel_switching_order::switch_channel_per_row, "Per Row (RZXCY)"},
	{ channel_switching_order::switch_channel_per_z, "Per Z (RXYCZ)"}
};

bool full_interface_gui::wrangle_scan()
{
	const auto mode = ui_->cmbAcquireModes->currentData().value<capture_mode>();
	auto& route = D->route;
	std::set<int> channels_used;
	//
	const auto basic_preflights = wrangle_capture_items(route, channels_used);
	const auto gui_based_preflights = [&]
	{
		std::vector<acquisition::preflight_function> preflight_checks;
		//
		const auto meta_data_preflight = [&] {
			return  ui_->metadata->toPlainText().isEmpty() ? QString("No experiment metadata provided") : QString();
		};
		preflight_checks.emplace_back(meta_data_preflight);
		//
		const auto huge_acquisition_preflight = [&]()
		{
			const auto item_count = route.cap.size();
			const auto bad = item_count >= acquisition::large_capture_threshold && route.filename_grouping == filename_grouping_mode::same_folder;
			return bad ? QString("Acquiring %1 in the same folder, prefer different file grouping").arg(item_count) : QString();
		};
		preflight_checks.emplace_back(huge_acquisition_preflight);
		//
		const auto not_enough_hard_drive_space = [&]()
		{
			const auto bad = ui_->prgStorage->maximum() == ui_->prgStorage->value();
			return bad ? QString("Not enough space at %1").arg(get_dir()) : QString();
		};
		preflight_checks.emplace_back(not_enough_hard_drive_space);
		//
		const auto predicate = [&](const acquisition::preflight_function& test)
		{
			return acquisition::prompt_if_failure(test());
		};
		const auto preflight_pass = std::all_of(preflight_checks.begin(), preflight_checks.end(), predicate);
		return preflight_pass;
	};
	const auto burst_preflights = [&]
	{
		const auto is_valid = (!capture_mode_settings::info.at(mode).is_burst) || route.is_valid_for_burst();
		if (!is_valid)
		{
			acquisition::prompt_if_failure("Acquisition is invalid or incompatible with burst mode");
			return false;
		}
		return is_valid;
	};
	const auto preflight_pass = basic_preflights && route.preflight_checks(channels_used).pass && gui_based_preflights() && burst_preflights();
	if (preflight_pass)
	{
		const auto file_path = QDir(get_dir()).absoluteFilePath(QString::fromStdString(full_interface_gui::default_scan_settings_name));
		save_cereal_file(file_path);

		save_metadata();
		start_acquisition(mode);
	}
	return preflight_pass;
}

bool full_interface_gui::wrangle_capture_items(acquisition& route, std::set<int>& channels_used)
{
	const auto rois = rois_->rowCount();
	if (rois < 1)
	{
		return false;
	}
	route.clear();
	route.ch = get_channel_settings();  //Only holds the channels in the compact light viewer
	route.filename_grouping = ui_->cmb_file_grouping->currentData().value<filename_grouping_mode>();
	const auto times = ui_->iterationTimes->value();
	for (auto time = 0; time < times; ++time)
	{
		//apply repeats and ROI delay after each repeat, move channel loop around the zee
		for (auto roi_index = 0; roi_index < rois; ++roi_index)
		{
			//include this little hack so if it takes too long the program doesn't outright crash
			qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
			//
			const auto item = rois_->get_serializable_item(roi_index);
			for (auto repeat_index = 0; repeat_index < item.repeats; ++repeat_index)
			{
				const auto cols = item.columns;
				const auto rows = item.rows;
				const auto column_step = item.column_step;
				const auto row_step = item.row_step;
				const auto z_stack_steps = item.pages;
				auto* roi_ptr = rois_->data_view_.at(roi_index);

				auto two_dee_tiles = [&]
				{
					static std::vector<cgal_triangulator::interpolation_pair> positions;
					positions.resize(0);
					const auto center = roi_ptr->get_grid_center();
					const auto x = center.x;
					const auto y = center.y;
					const auto from_tc = x - (cols * .5 - .5) * column_step;
					const auto from_tr = y - (rows * .5 - .5) * row_step;
					const auto n = snake_iterator::count(cols, rows);
					positions.reserve(n);
					for (auto i = 0; i < n; i++)
					{
						const auto snake = snake_iterator::iterate(i, cols);
						const auto x_position = from_tc + snake.column * column_step;
						const auto y_position = from_tr + snake.row * row_step;
						positions.emplace_back(snake, scope_location_xyz(x_position, y_position, std::numeric_limits<float>::quiet_NaN()));
					}

					roi_ptr->triangulator.interpolate(positions);
					return positions;
				}();
				auto push_new_item = [&](const cgal_triangulator::interpolation_pair& pair, const int zp, const int channel_idx)
				{
					const auto zee_offset = route.ch.at(channel_idx).z_offset;
					const auto x_position = pair.second.x, y_position = pair.second.y, zee_position = pair.second.z;
					const auto column = pair.first.column;
					const auto row = pair.first.row;
					const auto page = zp + z_stack_steps;
					const auto z = zee_offset + zee_position + zp * item.page_step;

					const scope_location_xyz loc(x_position, y_position, z);
					const roi_name roi_name(roi_index, time, repeat_index, column, row, page);
					const auto pos = capture_item(roi_name, scope_delays(), loc, channel_idx, false, scope_action::capture);
					route.cap.push_back(pos);
				};

				const auto switch_channels_per_roi = [&]()
				{
					auto flip_z = false;
					for (const auto& channel_idx : item.channels)		//channels that our specific ROI holds 
					{
						for (const auto& pair : two_dee_tiles)
						{
							for (auto zp = -z_stack_steps; zp <= z_stack_steps; zp++)
							{
								const auto zp_flip = flip_z ? (-1 * zp) : zp;
								push_new_item(pair, zp_flip, channel_idx);
							}
							flip_z = flip_z ? false : true;
						}
						channels_used.insert(channel_idx);
					}
				};
				const auto switch_channels_per_grid = [&]()
				{
					for (auto zp = -z_stack_steps; zp <= z_stack_steps; zp++)
					{
						for (const auto& channel_idx : item.channels)
						{
							for (const auto& pair : two_dee_tiles)
							{
								push_new_item(pair, zp, channel_idx);
							}
							channels_used.insert(channel_idx);
						}
					}
				};
				const auto switch_channels_per_tile = [&]()
				{
					auto flip_z = false;
					for (const auto& pair : two_dee_tiles)
					{
						for (auto zp = -z_stack_steps; zp <= z_stack_steps; zp++)
						{
							const auto zp_flip = flip_z ? (-1 * zp) : zp;
							for (const auto& channel_idx : item.channels)
							{
								push_new_item(pair, zp_flip, channel_idx);
								channels_used.insert(channel_idx);
							}
						}
						flip_z = flip_z ? false : true;
					}
				};
				const auto switch_channels_per_row = [&]()
				{
					for (auto zp = -z_stack_steps; zp <= z_stack_steps; zp++)
					{
						for (auto i = 0; i < rows; ++i)
						{
							for (const auto& channel_idx : item.channels)
							{
								for (auto j = 0; j < cols; ++j)
								{
									const auto& pair = two_dee_tiles.at((i * cols) + j);
									push_new_item(pair, zp, channel_idx);
								}
								channels_used.insert(channel_idx);
							}
						}
					}
				};

				const auto switch_channels_per_z = [&]()
				{
					auto flip_z = false;
					for (const auto& pair : two_dee_tiles)
					{
						for (const auto& channel_idx : item.channels)
						{
							for (auto zp = -z_stack_steps; zp <= z_stack_steps; zp++)
							{
								const auto zp_flip = flip_z ? (-1 * zp) : zp;
								push_new_item(pair, zp_flip, channel_idx);
							}
							flip_z = flip_z ? false : true;
							channels_used.insert(channel_idx);
						}
					}
				};

				const auto option = ui_->cmb_switch_channel->currentData().value<channel_switching_order>();
				switch (option)
				{
				case channel_switching_order::switch_channel_per_roi:
					switch_channels_per_roi();
					break;
				case channel_switching_order::switch_channel_per_grid:
					switch_channels_per_grid();
					break;
				case channel_switching_order::switch_channel_per_tile:
					switch_channels_per_tile();
					break;
				case channel_switching_order::switch_channel_per_row:
					switch_channels_per_row();
					break;
				case channel_switching_order::switch_channel_per_z:
					switch_channels_per_z();
					break;
				default:
					qli_not_implemented();
				}

				auto& last_item = route.cap.back();
				last_item.roi_move_delay = item.roi_move_delay;
				last_item.sync_io = item.io_sync_point;
			}
		}
	}
	route.cap.back().roi_move_delay = ms_to_chrono(0);
	route.cap.back().sync_io = false;
	//
	return true;
}


