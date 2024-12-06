#include "stdafx.h"
#include "compact_light_path_selector.h"
#include "device_factory.h"
#include "camera_device.h"
#include "qli_runtime_error.h"
#include "scope.h"
#include "ui_compact_light_path_selector.h"

void compact_light_path_selector::update_light_path()
{
	const auto current = get_compact_light_path();
	emit compact_light_path_changed(current);
}

compact_light_path_selector::~compact_light_path_selector()= default;

compact_light_path_selector::compact_light_path_selector(QWidget* parent) : QGroupBox(parent), id_(-1)
{
	ui_ = std::make_unique<Ui::compact_light_path_selector>();
	ui_->setupUi(this);

	setup_custom_name();
	//note signals like minimal exposure isn't connected (!)
	connect(ui_->btnRemove, &QPushButton::clicked, [&] {emit remove_me(id_); });
	const auto fixup_display_settings = [&](const processing_quad& quad)
	{
		const auto display_range = phase_processing_setting::settings.at(quad.processing);
		const display_settings display_settings(display_range.display_lut, display_range.display_range);
		ui_->wdg_display_selector->set_display_settings(display_settings);
		ui_->wdg_exposure_sequence->current_phase_retrieval_changed(quad.retrieval);
		update_light_path();
	};
	connect(ui_->wdg_exposure_sequence, &exposure_sequence_control::phase_shift_exposures_and_delays_changed, this, &compact_light_path_selector::update_light_path);
	connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, fixup_display_settings);
	connect(ui_->wdg_roi, &camera_config_selector::camera_config_changed, [&](const camera_config& config)
	{
		emit update_light_path();
		ui_->wdg_light_path->fixup_light_path(config.camera_idx);
	});
	connect(ui_->wdg_light_path, &light_path_selector::light_path_selector_changed, this, &compact_light_path_selector::update_light_path);
	connect(ui_->wdg_display_selector, &display_selector::display_settings_changed, this, &compact_light_path_selector::update_light_path);
	connect(ui_->qsb_zee_offset, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &compact_light_path_selector::update_light_path);
	connect(ui_->wdg_band_pass_filter, &band_pass_settings_selector::band_pass_settings_changed, this, &compact_light_path_selector::update_light_path);
}

compact_light_path compact_light_path_selector::get_compact_light_path() const
{
	const auto quad = ui_->processing_quad->get_quad();
	const auto light_path = ui_->wdg_light_path->get_light_path();
	const auto display = ui_->wdg_display_selector->get_display_settings();
	const auto zee = ui_->qsb_zee_offset->value();
	const auto frames = ui_->wdg_exposure_sequence->get_exposures_and_delays();
	const auto roi = ui_->wdg_roi->get_camera_config();
	const auto band_pass = ui_->wdg_band_pass_filter->get_band_pass_settings();

	const auto custom_label = ui_->channel_name->text();

	compact_light_path path(quad, light_path, display, roi, zee, frames, band_pass, custom_label.toStdString());
#if _DEBUG
	{
		if (!path.is_supported_quad())
		{
			qli_runtime_error("Invalid light path entered the widget, you suck");
		}
	}
#endif
	return path;
}

void compact_light_path_selector::set_compact_light_path(const compact_light_path& light_path) const
{
#if _DEBUG
	{
		if (!light_path.is_valid())
		{
			qli_invalid_arguments();
		}
	}
#endif
	const auto dirty_light_path_checkery_doo = [&]
	{
#if _DEBUG
		const auto now_its = ui_->wdg_light_path->get_light_path();
		if (!now_its.item_approx_equals(light_path))
		{
			qli_gui_mismatch();
		}
#endif
	};
	ui_->channel_name->setText(QString::fromStdString(light_path.label_suffix));
	ui_->wdg_exposure_sequence->set_phase_shift_exposures_and_delays(light_path.frames);
	ui_->processing_quad->set_processing(light_path);
	ui_->wdg_roi->set_camera_config(light_path);
	ui_->wdg_light_path->set_light_path_selector(light_path);
	dirty_light_path_checkery_doo();
	ui_->wdg_display_selector->set_display_settings(light_path);
	dirty_light_path_checkery_doo();
	dirty_light_path_checkery_doo();
	ui_->qsb_zee_offset->setValue(light_path.zee_offset);
	dirty_light_path_checkery_doo();
	dirty_light_path_checkery_doo();
	ui_->wdg_band_pass_filter->set_band_pass_settings(light_path);
	dirty_light_path_checkery_doo();

#if _DEBUG
	{
		const auto what_we_got = get_compact_light_path();
		if (!light_path.item_approx_equals(what_we_got))
		{
			qli_gui_mismatch();
		}
	}
#endif

}

void compact_light_path_selector::enable_buttons(const bool enable) const
{
	ui_->processing_quad->setEnabled(enable);
	ui_->wdg_light_path->setEnabled(enable);
	ui_->wdg_display_selector->setEnabled(enable);
	ui_->wdg_exposure_sequence->setEnabled(enable);
	ui_->wdg_band_pass_filter->setEnabled(enable); //Todo need this here? //When is this function even called? 
}

void compact_light_path_selector::set_id(const int id)
{
	id_ = id;
	const auto id_string = "Channel " + QString::number(id_);
	ui_->displayID->setText(id_string);
	const auto first = id == 0;
	//enabled instead of hidden for sizing reasons
	ui_->btnRemove->setEnabled(!first);
	ui_->qsb_zee_offset->setEnabled(!first);
}

void compact_light_path_selector::fixup_custom_name()
{
	const auto phase_retrieval = ui_->processing_quad->get_quad().retrieval;
	const auto& channel_names = D->scope->get_channel_settings_names();
	const auto fl_channel_index = ui_->wdg_light_path->get_light_path().scope_channel;
	const auto& channel_name = channel_names.at(fl_channel_index);
	const auto new_label = compute_and_scope_settings::fixup_label_suffix(phase_retrieval,channel_name.toStdString());
	const auto is_camera = phase_retrieval == phase_retrieval::camera;
	ui_->channel_name->setEnabled(is_camera);
	ui_->channel_name->setText(QString::fromStdString(new_label));
}

void compact_light_path_selector::setup_custom_name()
{
	fixup_custom_name();
	connect(ui_->wdg_light_path, &light_path_selector::light_path_selector_changed, this, &compact_light_path_selector::fixup_custom_name);
	connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, this, &compact_light_path_selector::fixup_custom_name);
}
