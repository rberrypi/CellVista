#include "stdafx.h"
#include "display_selector.h"
#include <QDoubleSpinBox>
#include "device_factory.h"
#include "qli_runtime_error.h"
#include "ui_display_selector.h"
struct name_and_icon
{
	QString name;
	QIcon icon;
};
static std::array<name_and_icon, display_settings::total_luts> lut_settings;
display_selector::display_selector(QWidget* parent) : QWidget(parent)
{
	if (lut_settings.front().name.isEmpty())
	{
		std::array<unsigned char,256*256*3> buffer ={0};
		const auto build_lut_label_and_icon = [&](const lut& item)
		{
			for (auto height = 0; height<256;++height)
			{
				for (auto width = 0 ;width<256;++width)
				{
					const auto idx = width+height*256;
					buffer.at(3*idx+0) = item.data.at(3*width+0);
					buffer.at(3*idx+1) = item.data.at(3*width+1);
					buffer.at(3*idx+2) = item.data.at(3*width+2);
				}
			}
			const QImage image(buffer.data(), 256, 256, QImage::Format_RGB888);
			const QIcon icon(QPixmap::fromImage(image));
			return name_and_icon{ QString::fromStdString(item.name),icon };
		};
		const auto& lut_info = display_settings::luts;
		std::transform(lut_info.begin(), lut_info.end(), lut_settings.begin(), build_lut_label_and_icon);
	}
	ui_ = std::make_unique<Ui::display_selector>();
	ui_->setupUi(this);
	ui_->gridLayout->setMargin(0);
	for (const auto& item : lut_settings)
	{
		ui_->cmbLUTs->addItem(item.icon, item.name);
	}
	for (auto&& lut : display_settings::luts)
	{
		QIcon item;
		const auto name = QString::fromStdString(lut.name);
	}
	connect(ui_->cmbLUTs, qOverload<int>(&QComboBox::currentIndexChanged), this, &display_selector::update_display_settings);
}

void display_selector::update_display_settings()
{
	const auto value = get_display_settings();
	emit display_settings_changed(value);
}

display_settings display_selector::get_display_settings() const
{
	const auto lut_idx = ui_->cmbLUTs->currentIndex();
	display_settings::display_ranges ranges;
	const auto box_count = min_boxes_.size();
	for (auto box_idx = 0; box_idx < box_count; ++box_idx)
	{
		const auto first = static_cast<float>(min_boxes_.at(box_idx)->value());
		const auto second = static_cast<float>(max_boxes_.at(box_idx)->value());
		ranges.push_back({ first,second });
	}
	return display_settings(lut_idx, ranges);
}

void display_selector::set_display_settings(const display_settings& settings)
{
	const auto shared = [&](QDoubleSpinBox* box, const float value)
	{
		const auto max_value = 999999;
		//box = new QDoubleSpinBox;
		box->setMinimum(-max_value);
		box->setMaximum(max_value);
		box->setAlignment(Qt::AlignHCenter);
		box->setButtonSymbols(QAbstractSpinBox::NoButtons);
		box->setSingleStep(0.1);
		box->setValue(value);
		connect(box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &display_selector::update_display_settings);
		return box;
	};
	const auto settings_to_set = settings.ranges.size();
	for (auto setting_idx = 0; setting_idx < settings_to_set; ++setting_idx)
	{
		const auto& range = settings.ranges.at(setting_idx);
		const auto current_boxes = min_boxes_.size();
		if (setting_idx >= current_boxes)
		{
			auto* min_box = shared(new QDoubleSpinBox, range.min);
			auto* max_box = shared(new QDoubleSpinBox, range.max);
			ui_->gridLayout->addWidget(min_box, setting_idx, 1);
			ui_->gridLayout->addWidget(max_box, setting_idx, 2);

			min_boxes_.push_back(min_box);
			max_boxes_.push_back(max_box);
		}
		else
		{
			min_boxes_.at(setting_idx)->setValue(range.min);
			max_boxes_.at(setting_idx)->setValue(range.max);
		}
	}
	const auto pop_widgets = [&](box_holder& holder)
	{
		while (holder.size() > settings_to_set)
		{
			auto* top = holder.back();
			holder.pop_back();
			ui_->gridLayout->removeWidget(top);
			top->setParent(nullptr);
			delete top;
		}
	};
	pop_widgets(min_boxes_);
	pop_widgets(max_boxes_);
	ui_->cmbLUTs->setCurrentIndex(settings.display_lut);
#if _DEBUG
	{
		const auto what_we_set = get_display_settings();
		if (!what_we_set.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
