#include "stdafx.h"
#include "darkfield_pattern_settings_selector.h"
#include "ui_darkfield_pattern_settings_selector.h"
#include "qli_runtime_error.h"
darkfield_pattern_settings_selector::darkfield_pattern_settings_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::darkfield_pattern_settings_selector>();
	ui_->setupUi(this);
	for (const auto& setting : darkfield_pattern_settings::darkfield_display_mode_settings)
	{
		const auto text = QString::fromStdString(setting.second);
		const auto data = QVariant::fromValue(setting.first);
		ui_->cmb_darkfield_show->addItem(text, data);
	}
	for (auto* double_spin_box : { ui_->sample_width ,ui_->ref_ring_na, ui_->objective_na ,ui_->max_na })
	{
		QObject::connect(double_spin_box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &darkfield_pattern_settings_selector::update_darkfield_settings);
	}
	for (auto* chk_box : { ui_->invert_modulator_x ,ui_->invert_modulator_y })
	{
		QObject::connect(chk_box, qOverload<int>(&QCheckBox::stateChanged), this, &darkfield_pattern_settings_selector::update_darkfield_settings);
	}
	QObject::connect(ui_->cmb_darkfield_show, qOverload<int>(&QComboBox::currentIndexChanged), this, &darkfield_pattern_settings_selector::update_darkfield_settings);
	set_complete(get_darkfield_pattern_settings().is_complete());
}

void darkfield_pattern_settings_selector::set_complete(const bool is_complete)
{
	const auto* color_label = is_complete ? "" : "color: red;";
	for (auto* item : { ui_->sample_width,ui_->ref_ring_na,ui_->objective_na,ui_->max_na })
	{
		item->setStyleSheet(color_label);
	}
}

void darkfield_pattern_settings_selector::update_darkfield_settings()
{
	const auto value = get_darkfield_pattern_settings();
	set_complete(value.is_complete());
	emit darkfield_pattern_settings_changed(value);
}

void darkfield_pattern_settings_selector::set_darkfield_pattern_settings_silent(const darkfield_pattern_settings& settings)
{
	QSignalBlocker blk(*this);
	set_darkfield_pattern_settings(settings);
}

void darkfield_pattern_settings_selector::set_darkfield_pattern_settings(const darkfield_pattern_settings& settings)
{
	ui_->sample_width->setValue(settings.width_na);
	ui_->ref_ring_na->setValue(settings.ref_ring_na);
	ui_->objective_na->setValue(settings.objective_na);
	ui_->max_na->setValue(settings.max_na);
	const auto display_mode_idx = ui_->cmb_darkfield_show->findData(QVariant::fromValue(settings.darkfield_display_mode));
	ui_->cmb_darkfield_show->setCurrentIndex(display_mode_idx);
	ui_->invert_modulator_x->setChecked(settings.invert_modulator_x);
	ui_->invert_modulator_y->setChecked(settings.invert_modulator_y);
#if _DEBUG
	{
		const auto what_we_got = get_darkfield_pattern_settings();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

darkfield_pattern_settings darkfield_pattern_settings_selector::get_darkfield_pattern_settings() const
{
	const auto width = ui_->sample_width->value();
	const auto ref_ring_na = ui_->ref_ring_na->value();
	const auto objective_na = ui_->objective_na->value();
	const auto max_na = ui_->max_na->value();
	const auto darkfield_align_mode = ui_->cmb_darkfield_show->currentData().value<darkfield_pattern_settings::darkfield_display_align_mode>();
	const auto invert_modulator_x = ui_->invert_modulator_x->isChecked();
	const auto invert_modulator_y = ui_->invert_modulator_y->isChecked();
	const darkfield_pattern_settings settings(width, ref_ring_na, objective_na, max_na, darkfield_align_mode, invert_modulator_x, invert_modulator_y);
	return settings;
}

void darkfield_pattern_settings_selector::enable_buttons(bool enable) const
{
	qli_not_implemented();
}
