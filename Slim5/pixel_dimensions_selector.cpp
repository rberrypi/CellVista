#include "stdafx.h"
#include "pixel_dimensions_selector.h"
#include "ui_pixel_dimensions_selector.h"
#include "qli_runtime_error.h"

void pixel_dimensions_selector::pixel_dimensions_update()
{
	const auto value = get_pixel_dimensions();
	const auto is_complete = value.is_complete();
	const auto* color_label = is_complete ? "" : "color: red;"; 
	ui_->coherence_length->setStyleSheet(color_label);
	ui_->pixel_ratio->setStyleSheet(color_label);	
	emit pixel_dimensions_changed(value);
}

pixel_dimensions_selector::pixel_dimensions_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::pixel_dimensions_selector>();
	ui_->setupUi(this);
	for (auto* wdg : {ui_->coherence_length,ui_->pixel_ratio})
	{
		QObject::connect(wdg,qOverload<double>(&QDoubleSpinBox::valueChanged), this, &pixel_dimensions_selector::pixel_dimensions_update);
	}
}

[[nodiscard]] pixel_dimensions pixel_dimensions_selector::get_pixel_dimensions() const
{
	const auto pixel_ratio = ui_->pixel_ratio->value();
	const auto coherence_length = ui_->coherence_length->value();
	return pixel_dimensions(coherence_length, pixel_ratio);
}

void pixel_dimensions_selector::set_pixel_dimensions(const pixel_dimensions& pixel_dimensions) const
{
	ui_->coherence_length->setValue(pixel_dimensions.coherence_length);
	ui_->pixel_ratio->setValue(pixel_dimensions.pixel_ratio);
#if _DEBUG
	{
		const auto what_we_got = get_pixel_dimensions();
		if (!what_we_got.item_approx_equals(pixel_dimensions))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
