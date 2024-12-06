#include "stdafx.h"
#include "ml_remapper_selector.h"

#include "render_settings.h"
#include "qli_runtime_error.h"
#include "ui_ml_remapper_selector.h"

void ml_remapper_selector::update_ml_remapper_selector()
{
	const auto values = get_ml_remapper();
	const auto is_doing_remapping = values.ml_remapper_type != ml_remapper_file::ml_remapper_types::off;
	ui->wdg_buttons->setEnabled(is_doing_remapping);
	ui->ml_display_settings->setEnabled(is_doing_remapping);
	emit ml_remapper_changed(values);
}

ml_remapper_selector::ml_remapper_selector(QWidget* parent) : QWidget(parent)
{
	ui = std::make_unique< Ui::ml_remapper_selector>();
	ui->setupUi(this);
	QStringList list_o_luts;
	for (auto&& lut : render_settings::luts)
	{
		list_o_luts << QString::fromStdString(lut.name);
	}
	ui->cmb_luts->addItems(list_o_luts);
	//
	QStringList remapper_files;
	for (const auto& item : ml_remapper_file::ml_remappers)
	{
		auto text = QString::fromStdString(item.second.network_label);
		auto data = QVariant::fromValue(item.first);
		ui->cmb_remapper_file->addItem(text, data);
	}
	//
	QObject::connect(ui->cmb_luts, qOverload<int>(&QComboBox::currentIndexChanged), this, &ml_remapper_selector::update_ml_remapper_selector);
	QObject::connect(ui->cmb_remapper_file, qOverload<int>(&QComboBox::currentIndexChanged), this, &ml_remapper_selector::update_ml_remapper_selector);
	for (auto item : { ui->min_range ,ui->max_range })
	{
		QObject::connect(item, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &ml_remapper_selector::update_ml_remapper_selector);
	}
}

ml_remapper ml_remapper_selector::get_ml_remapper() const
{
	const auto remapper_kind = ui->cmb_remapper_file->currentData().value< ml_remapper_file::ml_remapper_types>();
	const auto min_value = static_cast<float>(ui->min_range->value());
	const auto max_value = static_cast<float>(ui->max_range->value());
	const display_range ml_display_range = { min_value,max_value };
	const auto lut_idx = ui->cmb_luts->currentIndex();
	const auto mode = [&]
	{
		if (ui->btn_overlay->isChecked())
		{
			return ml_remapper::display_mode::overlay;
		}
		if (ui->btn_transform_only->isChecked())
		{
			return ml_remapper::display_mode::only_remap;
		}
		return ml_remapper::display_mode::only_phase;
	}();
	const ml_remapper ml_remapper(remapper_kind, ml_display_range, lut_idx, mode);
	return ml_remapper;
}

void ml_remapper_selector::set_ml_remapper(const ml_remapper& remapper) const
{
	ui->cmb_luts->setCurrentIndex(remapper.ml_lut);
	const auto idx = ui->cmb_remapper_file->findData(QVariant::fromValue(remapper.ml_remapper_type));
	ui->cmb_remapper_file->setCurrentIndex(idx);
	ui->min_range->setValue(remapper.ml_display_range.min);
	ui->max_range->setValue(remapper.ml_display_range.max);
	switch (remapper.ml_display_mode)
	{
	case  ml_remapper::display_mode::only_remap:
		ui->btn_transform_only->setChecked(true);
		break;
	case  ml_remapper::display_mode::only_phase:
		ui->btn_phase_only->setChecked(true);
		break;		
	case  ml_remapper::display_mode::overlay:
		ui->btn_transform_only->setChecked(true);
		break;
	default:
		qli_runtime_error("Not Implemented");
	}
#if _DEBUG
	{
		const auto what_we_got = get_ml_remapper();
		if (!what_we_got.item_approx_equals(remapper))
		{
			qli_runtime_error("Some GUI Error");
		}
	}
#endif
}

