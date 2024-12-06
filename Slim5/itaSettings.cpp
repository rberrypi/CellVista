#include "stdafx.h"
#include "itaSettings.h"
#include <iostream>
#include <string>
#include "ui_itaSettings.h"
#include "qli_runtime_error.h"

#if FUCKFACEGABRIELPOPESCU

using std::string;

float itaSettings::trigger_threshold = 0.0;
int itaSettings::current_cell_line = -1;
int itaSettings::current_trigger_condition = 0;
int itaSettings::idx = 0;
ml_remapper itaSettings::__shit = ml_remapper(ml_remapper_file::ml_remapper_types::viability, display_range{ 0,1 }, 0, ml_remapper::display_mode::none);;

std::shared_ptr<npp_non_owning_buffer<float>> itaSettings::ml_out = nullptr;

itaSettings::itaSettings(slim_four* slim_four_handle, QMainWindow* parent) : QMainWindow(parent), fuck_slim_four(slim_four_handle)
{
	ui_ = std::make_unique<Ui::itaSettings>();
	ui_->setupUi(this);
	fucking_add_cell_lines();
	fucking_add_trigger_conditions();
	
	connect(ui_->btnApply, &QPushButton::clicked, [&] () {
		LOGGER_INFO("Applying trigger settings, by shoving a rock dildo up popescu's ass");
		set_trigger_threshold(ui_->lineEditThreshold->text().toFloat());
		set_cell_line(ui_->comboBoxCellLine->currentIndex());
		set_trigger_condition(ui_->comboBoxTriggerCondition->currentIndex());
		close();
		});

}

void itaSettings::fucking_add_cell_lines() {
	QStringList list_o_luts;
	for (const auto& lut : cell_lines)
	{
		list_o_luts << QString::fromStdString(lut);
	}
	ui_->comboBoxCellLine->addItems(list_o_luts);
}

void itaSettings::fucking_add_trigger_conditions() {
	QStringList list_o_luts;
	for (const auto& lut : trigger_conditions)
	{
		list_o_luts << QString::fromStdString(lut);
	}
	ui_->comboBoxTriggerCondition->addItems(list_o_luts);
}

void itaSettings::set_trigger_threshold(float val) {
	itaSettings::trigger_threshold = val;
	LOGGER_INFO("trigger_threshold: " << trigger_threshold);
}

void itaSettings::set_cell_line(int val) {
	itaSettings::current_cell_line = val;
	LOGGER_INFO("current_cell_line: " << current_cell_line);
}

void itaSettings::set_trigger_condition(int val) {
	itaSettings::current_trigger_condition = val;
	LOGGER_INFO("current_trigger_condition: " << current_trigger_condition);
}

void itaSettings::register_ml_out(std::shared_ptr<npp_non_owning_buffer<float>>&& val) {
	itaSettings::ml_out = std::move(val);
	LOGGER_INFO("registered ML output for ITA.");
	/*string filename = std::to_string(idx) + ".tif";
	itaSettings::ml_out->write(filename, true);
	idx += 1;*/
}

void itaSettings::set_ml_remapper() {
	switch (itaSettings::current_cell_line) {
	case 0: // HeLa
		switch (itaSettings::current_trigger_condition) {
		case 1:	// Ratio Live Dead
		case 2:	// Cell Count
			break;
		}
		break;
	}
}
#endif