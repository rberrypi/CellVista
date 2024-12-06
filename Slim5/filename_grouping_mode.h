#pragma once
#ifndef FILENAME_GROUPING_MODE_H
#define FILENAME_GROUPING_MODE_H
#include <unordered_map>
enum class filename_grouping_mode { same_folder, fov_channel };
const extern std::unordered_map<filename_grouping_mode, std::string> filename_grouping_names;

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(filename_grouping_mode)
#endif
#endif