#include "stdafx.h"
#include "camera_config_selector.h"
#include "device_factory.h"
#include "camera_device.h"
#include "ui_camera_aoi_selector.h"
#include "qli_runtime_error.h"

camera_config_selector::camera_config_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::camera_aoi_selector>();
	ui_->setupUi(this);
	//
	const auto camera_sizes = D->cameras.size();
	auto char_length_hack = 0;
	for (auto camera_idx = 0; camera_idx < camera_sizes; ++camera_idx)
	{
		auto&& camera = D->cameras.at(camera_idx);
		auto modes = camera->get_aoi_names();
		for (auto aoi_index = 0; aoi_index < modes.size(); ++aoi_index)
		{
			const camera_config_aoi_camera_pair pair(aoi_index, camera_idx);
			const auto user_data = QVariant::fromValue(pair);
			const auto& str = modes.at(aoi_index);
			char_length_hack = std::max(str.size(), char_length_hack);
			ui_->cmbcamera_aoi->addItem(str, user_data);
		}
	}
	ui_->cmbcamera_aoi->setMinimumContentsLength(char_length_hack);
	connect(ui_->cmbcamera_aoi, qOverload<int>(&QComboBox::currentIndexChanged), this, &camera_config_selector::update_modes);
	for (auto* cmb : { ui_->cmbcamera_aoi,ui_->cmbcamera_gain,ui_->cmbcamera_bin })
	{
		connect(cmb, qOverload<int>(&QComboBox::currentIndexChanged), this, &camera_config_selector::camera_config_update);
	}
	update_modes();
}

void camera_config_selector::camera_config_update()
{
	const auto values = get_camera_config();
	emit camera_config_changed(values);
}

void camera_config_selector::update_modes()
{
	//should really be its own class as we often want this kind of behavior
	const auto safe_add_items = [](QComboBox* box, const QStringList& items)
	{
		{
			QSignalBlocker blk(box);
			const auto old_data = box->currentText();
			box->clear();
			box->addItems(items);
			const auto index = box->findText(old_data);
			if (index >= 0)
			{
				box->setCurrentIndex(index);
			}
		}
		box->setHidden(items.size() < 2);
	};
	const auto pair = ui_->cmbcamera_aoi->currentData().value<camera_config_aoi_camera_pair>();
	auto& camera = D->cameras.at(pair.camera_idx);
	const auto gain_modes = camera->get_gain_names();
	safe_add_items(ui_->cmbcamera_gain, gain_modes);
	const auto bin_modes = camera->get_bin_names();
	safe_add_items(ui_->cmbcamera_bin, bin_modes);
}

camera_config camera_config_selector::get_camera_config() const
{
	const auto data = ui_->cmbcamera_aoi->currentData();
	const auto cfg = data.value<camera_config_aoi_camera_pair>();
	const auto gain_idx = ui_->cmbcamera_gain->currentIndex();
	const auto bin_idx = ui_->cmbcamera_bin->currentIndex();
	const auto cooling = true;
	const camera_config config(gain_idx, bin_idx, cfg.aoi_index, camera_mode::software, cooling, cfg.camera_idx);
	return config;
}

void camera_config_selector::set_camera_config(const camera_config& mode) const
{
	{
		auto idx = 0;
		for (auto i = 0; i < ui_->cmbcamera_aoi->count(); ++i)
		{
			const auto current = ui_->cmbcamera_aoi->itemData(i).value<camera_config_aoi_camera_pair>();
			if (current.camera_idx == mode.camera_idx && current.aoi_index == mode.aoi_index)
			{
				idx = i;
				break;
			}
		}
		if (idx >= 0)
		{
			ui_->cmbcamera_aoi->setCurrentIndex(idx);
		}
		else
		{
			qli_gui_mismatch();
		}
	}
	//
	ui_->cmbcamera_gain->setCurrentIndex(mode.gain_index);
	ui_->cmbcamera_bin->setCurrentIndex(mode.bin_index);

#if _DEBUG
	{
		const auto what_we_got = get_camera_config();
		if (!(what_we_got == mode))
		{
			qli_gui_mismatch();
		}
	}
#endif

}
