#include "stdafx.h"
#include "segmentation_editor.h"

#include "qli_runtime_error.h"
#include "ui_segmentation_editor.h"

void segmentation_editor::update_segmentation_settings() 
{
	const auto value = get_segmentation();
	emit segmentation_changed(value);
}

segmentation_editor::segmentation_editor(const segmentation_settings& settings, QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::SegmentationEditor>();
	ui_->setupUi(this);
	for (auto&& item : threshold)
	{
		const auto mode = QVariant::fromValue(item.first);
		ui_->cmbsegmentation_mode->addItem(QString::fromStdString(item.second), mode);
	}
	set_segmentation(settings);
	//
	const auto get_n_signal_functor = [&] {emit segmentation_changed(get_segmentation()); };
	//YES THE CAPTURE GROUP TYPE IS IMPORTANT
	connect(ui_->cmbsegmentation_mode, static_cast<void (QComboBox ::*)(int)>(&QComboBox::currentIndexChanged), [=](int)
	{
		get_n_signal_functor();
		const auto mode = ui_->cmbsegmentation_mode->currentData().value<segmentation_mode>();
		const auto buttons = mode != segmentation_mode::off;
		enable_buttons(buttons);
	});
	connect(ui_->qsbSegmentationThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMinArea, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMaxArea, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMinBounds, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMaxBounds, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMinCirc, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->qsbSegmentationMaxCirc, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->chkKeepOriginals, &QCheckBox::stateChanged, this, &segmentation_editor::update_segmentation_settings);
	connect(ui_->btnResetSegmentation, &QPushButton::pressed, [&] {set_segmentation(segmentation_settings::default_segmentation_settings()); });
}

segmentation_editor::~segmentation_editor() = default;

segmentation_settings segmentation_editor::get_segmentation() const
{
	const auto mode = ui_->cmbsegmentation_mode->currentData();
#if _DEBUG
	if (!mode.canConvert<segmentation_mode>())
	{
		qli_runtime_error("you suck");
	}
#endif
	const auto mode_as_mode = mode.value<segmentation_mode>();
	const auto  threshold = ui_->qsbSegmentationThreshold->value();
	const segmentation_feature_area area_feature = { static_cast<float>(ui_->qsbSegmentationMinArea->value()),static_cast<float>(ui_->qsbSegmentationMaxArea->value()) };
	const segmentation_feature_bounding bounding_feature = { static_cast<float>(ui_->qsbSegmentationMinBounds->value()), static_cast<float>(ui_->qsbSegmentationMaxBounds->value()) };
	const segmentation_feature_circularity circularity_feature = { static_cast<float>(ui_->qsbSegmentationMinCirc->value()), static_cast<float>(ui_->qsbSegmentationMaxCirc->value()) };
	const segmentation_save_settings save_settings = { ui_->chkKeepOriginals->isChecked() };
	return segmentation_settings(mode_as_mode, threshold, bounding_feature, circularity_feature, area_feature, save_settings);
}

void segmentation_editor::enable_buttons(const bool enable) const
{
	ui_->grbSaveButtons->setEnabled(enable);
	ui_->wdgSegmentationButtons->setEnabled(enable);
}

void segmentation_editor::set_segmentation(const segmentation_settings & settings) const
{
	const auto buttons = settings.segmentation != segmentation_mode::off;
	if (buttons)
	{
		enable_buttons(buttons);
	}
	const auto as_variant = QVariant::fromValue(settings.segmentation);
	const auto idx = ui_->cmbsegmentation_mode->findData(as_variant);
	ui_->cmbsegmentation_mode->setCurrentIndex(idx);
#if _DEBUG
	if (idx != ui_->cmbsegmentation_mode->currentIndex())
	{
		qli_runtime_error("you suck");
	}
#endif
	ui_->qsbSegmentationThreshold->setValue(settings.segmentation_min_value);
	ui_->qsbSegmentationMinArea->setValue(settings.segmentation_area_min);
	ui_->qsbSegmentationMaxArea->setValue(settings.segmentation_area_max);
	ui_->qsbSegmentationMinBounds->setValue(settings.segmentation_bounding_min);
	ui_->qsbSegmentationMaxBounds->setValue(settings.segmentation_bounding_max);
	ui_->qsbSegmentationMinCirc->setValue(settings.segmentation_circ_min);
	ui_->qsbSegmentationMaxCirc->setValue(settings.segmentation_circ_max);
	ui_->chkKeepOriginals->setCheckState(settings.segmentation_keep_originals ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
	if (!buttons)
	{
		enable_buttons(buttons);
	}
#if _DEBUG
	{
		const auto what_we_got = get_segmentation();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
