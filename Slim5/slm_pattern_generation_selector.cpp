#include "stdafx.h"
#include "slm_pattern_generation_selector.h"

#include <QStandardItemModel>


#include "phase_processing.h"
#include "ui_slm_pattern_generation_selector.h"
#include "qli_runtime_error.h"

void slm_pattern_generation_selector::update_slm_pattern_generation()
{
	const auto values = get_slm_pattern_generation();
	const auto show_darkfield = slm_mode::darkfield == values.modulator_mode;
	ui_->cmb_darkfield->setEnabled(show_darkfield);
	ui_->darkfield_samples->setEnabled(show_darkfield);
	emit slm_pattern_generation_changed(values);
}

slm_pattern_generation_selector::slm_pattern_generation_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::slm_pattern_generation_selector>();
	ui_->setupUi(this);
	for (auto mode : { slm_mode::single_shot,slm_mode::two_shot_lcvr, slm_mode::slim, slm_mode::qdic, slm_mode::darkfield, })
	{
		const auto& label = slm_mode_setting::settings.at(mode).label;
		ui_->slm_mode->addItem(QString::fromStdString(label), QVariant::fromValue(mode));
	}
	for (const auto& item : darkfield_mode_settings::settings)
	{
		const auto& label = item.second;
		ui_->cmb_darkfield->addItem(QString::fromStdString(label.label), QVariant::fromValue(item.first));
	}
	for (auto cmb : { ui_->slm_mode,ui_->cmb_darkfield })
	{
		QObject::connect(cmb, qOverload<int>(&QComboBox::currentIndexChanged), this, &slm_pattern_generation_selector::update_slm_pattern_generation);
	}
	ui_->cmb_darkfield->setEnabled(false);
	QObject::connect(ui_->darkfield_samples, qOverload<int>(&QSpinBox::valueChanged), this, &slm_pattern_generation_selector::update_slm_pattern_generation);
}

slm_pattern_generation slm_pattern_generation_selector::get_slm_pattern_generation() const
{
	const auto slm = ui_->slm_mode->currentData().value<slm_mode>();
	const auto darkfield = ui_->cmb_darkfield->currentData().value<darkfield_mode>();
	const auto samples = ui_->darkfield_samples->value();
	return slm_pattern_generation(slm, darkfield, samples);
}

void slm_pattern_generation_selector::set_processing_double(const processing_double& processing)
{
	auto* model = qobject_cast<QStandardItemModel*>(ui_->slm_mode->model());
	
	const auto count = model->rowCount();
	for (auto i = 0; i < count; ++i)
	{
		const auto mode = ui_->slm_mode->itemData(i).value<slm_mode>();
		const auto required_patterns = phase_retrieval_setting::settings.at(processing.retrieval).modulator_patterns();
		const auto sufficient_number_of_patterns = slm_mode_setting::settings.at(mode).patterns >= required_patterns;
		const auto enabled = required_patterns == pattern_count_from_file ? true : sufficient_number_of_patterns;
		model->item(i)->setEnabled(enabled);
	}
}

void slm_pattern_generation_selector::set_slm_pattern_generation_silent(const slm_pattern_generation& settings)
{
	QSignalBlocker sb(*this);
	set_slm_pattern_generation(settings);
}

void slm_pattern_generation_selector::set_slm_pattern_generation(const slm_pattern_generation& settings) const
{
	const auto slm_idx = ui_->slm_mode->findData(QVariant::fromValue(settings.modulator_mode));
	ui_->slm_mode->setCurrentIndex(slm_idx);
	const auto darkfield_idx = ui_->cmb_darkfield->findData(QVariant::fromValue(settings.darkfield));
	ui_->cmb_darkfield->setCurrentIndex(darkfield_idx);
	ui_->darkfield_samples->setValue(settings.darkfield_samples);
#if _DEBUG
	{
		const auto what_we_got = get_slm_pattern_generation();
		if (!(what_we_got == settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
