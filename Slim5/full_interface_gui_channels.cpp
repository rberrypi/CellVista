#include "stdafx.h"
#include "full_interface_gui.h"
#include "ui_full_interface_gui.h"
void full_interface_gui::setup_common_channels()
{
	connect(ui_->btnSetOne, &QPushButton::clicked, [&] {add_common_channel(0); });
	connect(ui_->btnSetTwo, &QPushButton::clicked, [&] {add_common_channel(1); });
	connect(ui_->btnSetThree, &QPushButton::clicked, [&] {add_common_channel(2); });
	connect(ui_->btnSetFour, &QPushButton::clicked, [&] {add_common_channel(3); });
	connect(ui_->btnSetFive, &QPushButton::clicked, [&] {add_common_channel(4); });
	connect(ui_->btnSetSix, &QPushButton::clicked, [&] {add_common_channel(5); });
	connect(ui_->btnSetSeven, &QPushButton::clicked, [&] {add_common_channel(6); });
	connect(ui_->btnSetEight, &QPushButton::clicked, [&] {add_common_channel(7); });
	connect(ui_->btnSetNine, &QPushButton::clicked, [&] {add_common_channel(8); });
}