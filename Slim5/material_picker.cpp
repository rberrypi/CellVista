#include "stdafx.h"
#include "material_picker.h"

#include "ui_material_picker.h"
#include "phase_processing.h"

const std::unordered_map<phase_processing, material_covers> material_picker::covers = std::unordered_map<phase_processing, material_covers>{
	{ phase_processing::mass,{ QString("dry mass density (fg)"), true, false, false, true, false } },
	{ phase_processing::height,{ QString("height (microns)"), true, true, true, false, false } },
	{ phase_processing::refractive_index,{ QString("refractive index (N_media + N_object)"), true, true, false, false, true } },
	{ phase_processing::mutual_intensity,{ QString("mutual intensity ([0,1])"), false, false, false, false, false } }
};

const material_covers material_picker::unsupported_cover = { QString(""), false, false, false, false, false };

void material_picker::set_material_info(const material_info& info) const
{

	ui->qsbHeight->setValue(info.obj_height);
	ui->qsbMassInc->setValue(info.mass_inc);
	ui->qsbRefCell->setValue(info.n_cell);
	ui->qsbRefMedia->setValue(info.n_media);
}

material_picker::material_picker(const phase_processing& processing, const material_info& info, QWidget* parent) : QMainWindow(parent)
{
	ui = std::make_unique<Ui::material_picker>();
	ui->setupUi(this);
	setAttribute(Qt::WA_DeleteOnClose, true);
	set_phase_processing(processing);
	set_material_info(info);
	for (auto* items : { ui->qsbRefMedia,ui->qsbRefCell,ui->qsbMassInc,ui->qsbHeight })
	{
		connect(items, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &material_picker::material_info_update);
	}
}

material_info material_picker::get_material_info() const
{
	const auto no = ui->qsbRefMedia->value();
	const auto nc = ui->qsbRefCell->value();
	const auto inc = ui->qsbMassInc->value();
	const auto height = ui->qsbHeight->value();
	const material_info info(no, nc, inc, height);
	return info;
}


void material_picker::material_info_update() const
{
	const auto info = get_material_info();
	emit material_info_updated(info);
}

void material_picker::set_phase_processing(const phase_processing idx) const
{
	const auto iterator = covers.find(idx);
	const auto valid = iterator != covers.end();
	const auto set = valid ? iterator->second : unsupported_cover;
	const auto title = valid ? QString("Measuring %1").arg(set.name) : set.name;
	ui->grbWhat->setTitle(title);
	ui->qsbHeight->setEnabled(set.show_microns);
	ui->qsbMassInc->setEnabled(set.show_increment);
	ui->qsbRefCell->setEnabled(set.show_obj);
	ui->qsbRefMedia->setEnabled(set.show_media);
}
