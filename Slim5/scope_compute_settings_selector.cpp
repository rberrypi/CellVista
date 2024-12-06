#include "stdafx.h"
#include "scope_compute_settings_selector.h"
#include "device_factory.h"
#include "ui_scope_compute_settings_selector.h"
#include "qli_runtime_error.h"
scope_compute_settings_selector::scope_compute_settings_selector(QWidget *parent) : QWidget(parent)
{
	ui_=std::make_unique<Ui::scope_compute_settings_selector>();
	ui_->setupUi(this);
	wavelength_widgets = { ui_->qsb_red,ui_->qsb_green,ui_->qsb_blue };
	for (auto* widgets : wavelength_widgets)
	{
		QObject::connect(widgets, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &scope_compute_settings_selector::update_scope_compute_settings);
	}
	auto other_widgets = { ui_->qsb_qdic_shear_angle, ui_->qsb_qdic_shear_dx ,ui_->objective_attenuation ,ui_->stage_overlap };
	for (auto* widgets : other_widgets)
	{
		QObject::connect(widgets, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &scope_compute_settings_selector::update_scope_compute_settings);
	}
	connect(ui_->compute_dimensions,&pixel_dimensions_selector::pixel_dimensions_changed,this,&scope_compute_settings_selector::update_scope_compute_settings);
	set_color(D->has_a_color_camera());
}

void scope_compute_settings_selector::set_color(const bool color)
{
	const auto is_grayscale = !color;
	ui_->qsb_green->setHidden(is_grayscale);
	ui_->qsb_blue->setHidden(is_grayscale);
}

void scope_compute_settings_selector::update_scope_compute_settings()
{
	const auto value = get_scope_compute_settings();
	const auto is_complete = value.is_complete();
	const auto* color_label = is_complete ? "" : "color: red;"; 
	ui_->objective_attenuation->setStyleSheet(color_label);
	ui_->stage_overlap->setStyleSheet(color_label);
	emit scope_compute_settings_changed(value);
}

scope_compute_settings scope_compute_settings_selector::get_scope_compute_settings() const
{
	//ui_->compute_dimensions->pixel
	const auto scope_compute_dimensions = ui_->compute_dimensions->get_pixel_dimensions();
	const float qsb_qdic_shear_angle = ui_->qsb_qdic_shear_angle->value();
	const float qsb_qdic_shear_dx = ui_->qsb_qdic_shear_dx->value();
	const qdic_scope_settings qdic_scope_settings(qsb_qdic_shear_angle, qsb_qdic_shear_dx);
	const auto objective_attenuation = ui_->objective_attenuation->value();
	//const auto stage_overlap = ui_->stage_overlap->value();
	const auto stage_overlap = 1.0-ui_->stage_overlap->value() / 100.0;
	const auto wave_length_package = get_wave_length_package();
	const scope_compute_settings settings(objective_attenuation, stage_overlap, scope_compute_dimensions, qdic_scope_settings, wave_length_package);
	return settings;
}

void scope_compute_settings_selector::set_scope_compute_settings_silent(const scope_compute_settings& settings)
{
	QSignalBlocker sb(*this);
	set_scope_compute_settings(settings);
}

void scope_compute_settings_selector::set_scope_compute_settings(const scope_compute_settings& settings)
{
	ui_->compute_dimensions->set_pixel_dimensions(settings);
	ui_->qsb_qdic_shear_angle->setValue(settings.qsb_qdic_shear_angle);
	ui_->qsb_qdic_shear_dx->setValue(settings.qsb_qdic_shear_dx);
	ui_->objective_attenuation->setValue(settings.objective_attenuation);
	const auto stage_overlap_display = -100.0*(settings.stage_overlap-1.0);
	ui_->stage_overlap->setValue(stage_overlap_display);
	set_wave_length_package(settings.wave_lengths);
#if _DEBUG
	{
		const auto what_we_set = get_scope_compute_settings();
		if (!what_we_set.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void scope_compute_settings_selector::set_wave_length_package(const wave_length_package& package)
{
	for (auto i = 0; i < package.size(); ++i)
	{
		const auto wavelength = package.at(i);
		wavelength_widgets.at(i)->setValue(wavelength);
	}
}

wave_length_package scope_compute_settings_selector::get_wave_length_package() const
{
	wave_length_package package;
	const auto functor = [](QDoubleSpinBox* spin_box)
	{
		return spin_box->value();
	};
	std::transform(wavelength_widgets.begin(), wavelength_widgets.end(), package.begin(), functor);
	return package;
}

