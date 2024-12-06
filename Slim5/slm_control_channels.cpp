#include "stdafx.h"
#include "slm_control.h"
#include "device_factory.h"
#include "scope.h"
#include "ui_slm_control.h"
void slm_control::setup_io_settings() const
{
	ui_->chkDisplayacquisition->setCheckState(D->io_show_files ? Qt::Checked : Qt::Unchecked);
	connect(ui_->chkDisplayacquisition, &QCheckBox::stateChanged, this, [&](const int val) { D->io_show_files = val == Qt::Checked; });
	ui_->chkDisplayProgress->setCheckState(D->io_show_cmd_progress ? Qt::Checked : Qt::Unchecked);
	connect(ui_->chkDisplayProgress, &QCheckBox::stateChanged, this, [&](const int val) { D->io_show_cmd_progress = val == Qt::Checked; });
}
void slm_control::setup_phase_channel_selection() const
{
	//setWindowFlags(this->windowFlags() | Qt::MSWindowsFixedSizeDialogHint);
	//statusBar()->setSizeGripEnabled(false);
	auto names = D->scope->get_channel_settings_names();
	const auto find_and_remove = [&](const std::string& str)
	{
		const auto match_me = QString::fromStdString(str);
		const auto it = std::find(names.begin(), names.end(), match_me);
		if (it != names.end())
		{
			const auto pos = it - names.begin();
			names.removeAt(pos);
		}
	};
	find_and_remove(scope_channel_drive::channel_off_str);
	find_and_remove(scope_channel_drive::channel_phase_str);
	ui_->cmbChannel->addItems(names);
	const auto chan = D->scope->chan_drive->phase_channel_alias;
	{
		ui_->cmbChannel->setCurrentIndex(chan);
	}
	connect(ui_->cmbChannel, qOverload<int>(&QComboBox::currentIndexChanged),[&](const int channel)
	{
			D->scope->chan_drive->phase_channel_alias = channel;
	});
	const auto is_transmission = D->scope->chan_drive->is_transmission;
	ui_->chkTransmission->setCheckState(is_transmission ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
	connect(ui_->chkTransmission, &QCheckBox::stateChanged, [](const int state)
	{
		D->scope->chan_drive->is_transmission = Qt::Checked == state;
	});
	ui_->qsbTimeout->setValue(to_mili(D->scope->chan_drive->channel_off_threshold));
	connect(ui_->qsbTimeout, qOverload<double>(&QDoubleSpinBox::valueChanged), [&] (const double value)
	{
		D->scope->chan_drive->channel_off_threshold = ms_to_chrono( value );
	});
}