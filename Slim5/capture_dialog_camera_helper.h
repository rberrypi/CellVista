#pragma once
#ifndef CAPTURE_DIALOG_CAMERA_HELPER
#define CAPTURE_DIALOG_CAMERA_HELPER
// ReSharper disable once CppInconsistentNaming
class QComboBox;
class QTableView;
class camera_device;
#include <vector>
void camera_settings_to_combobox(QComboBox* combobox,const std::vector<camera_device*>& cameras);

void style_table_view(QTableView* view);
#endif