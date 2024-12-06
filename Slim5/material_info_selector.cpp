#include "stdafx.h"
#include "material_info_selector.h"
#include "ui_material_info_selector.h"
#include "qli_runtime_error.h"

material_info_selector::material_info_selector(QWidget *parent) : QWidget(parent)
{
	ui_=std::make_unique<Ui::material_info_selector>();
	ui_->setupUi(this);
	for (auto* widgets : { ui_->n_media,ui_->n_cell,ui_->mass_inc,ui_->obj_height })
	{
		QObject::connect(widgets, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &material_info_selector::update_material_info);
	}
}

void  material_info_selector::update_material_info()
{
	const auto value = get_material_info();
	emit material_settings_changed(value);
}

material_info material_info_selector::get_material_info() const
{
	const auto n_media = ui_->n_media->value();
	const auto n_cell = ui_->n_cell->value();
	const auto mass_inc = ui_->mass_inc->value();
	const auto obj_height = ui_->obj_height->value();
	const auto info = material_info(n_media, n_cell, mass_inc, obj_height);
	return info;
}

void material_info_selector::set_material_info(const material_info& settings)const
{
	ui_->n_cell->setValue(settings.n_cell);
	ui_->n_media->setValue(settings.n_media);
	ui_->mass_inc->setValue(settings.mass_inc);
	ui_->obj_height->setValue(settings.obj_height);
#if _DEBUG
	{
		const auto what_we_set = get_material_info();
		if (!what_we_set.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}