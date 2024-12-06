#pragma once
#include "program_config.h"

#ifndef ITA_SETTINGS_
#define ITA_SETTINGS_
#if FUCKFACEGABRIELPOPESCU
#include <npp_non_owning_buffer.h>
#include <QMainWindow>
#include <vector>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include "slim_four.h"
#include <ml_structs.h>

using std::pair;
using std::vector;
using std::unordered_map;
using std::vector;
using std::string;

namespace Ui {
	class itaSettings;
}

// fuck gabriel popescu. The guy has no clue as to what he wants. I think it's time he retires, after all
// he is 50. I think his age is starting to take a hold on his mental capacity to function as a supervisor.
// Honestly, this guy knows less than the average user. This guy has no clue about what's in the software
// eventhough it literally was the project of one of his PhD students. Frankly, at this point, the guy is an
// embarassment - not only as a supervisor, but to the entire homosapien species. This guy makes me feel
// disgusted to call myself human.

// If you're a PhD, and the guy is making you work on this software shit, I think you should consider
// changing supervisors.

// helpful links:
//   shorturl.at/bmowO

class itaSettings final : public QMainWindow
{

	Q_OBJECT;

	std::unique_ptr<Ui::itaSettings> ui_;

	vector<string> trigger_conditions = {
		"Off",
		"Ratio Live Dead", // viability model
		"Cell Count",	// viability model
		"Confluency"	// need new model
	};

	vector<string> cell_lines = {
		"HeLa",
		"CHO"
	};

public:
	static ml_remapper __shit;
	static int idx;

	static std::shared_ptr<npp_non_owning_buffer<float>> ml_out;
	slim_four* fuck_slim_four;
	static int current_cell_line;
	static int current_trigger_condition;
	static float trigger_threshold;

	explicit itaSettings(slim_four* slim_four_handle, QMainWindow* parent);

	void fucking_add_cell_lines();
	void fucking_add_trigger_conditions();
	void set_trigger_threshold(float val);
	void set_cell_line(int val);
	void set_trigger_condition(int val);
	static void set_ml_remapper();

	static void register_ml_out(std::shared_ptr<npp_non_owning_buffer<float>>&& val);

	template<typename T>
	static bool live_dead_trigger(const T* data, unsigned int cols, unsigned int rows, int samples_per_pixel) {
		LOGGER_INFO("live_dead_trigger ita called");
		itaSettings::set_ml_remapper();
		return true;
	}

	template<typename T>
	static bool cell_count_trigger(const T* data, unsigned int cols, unsigned int rows, int samples_per_pixel) {
		LOGGER_INFO("cell_count_trigger ita called");
		LOGGER_INFO("size in bytes: " << sizeof(T)*cols*rows*samples_per_pixel);
		/*vector<vector<vector<t>>> data = { vector<vector<t>> , vector<vector<t>> , vector<vector<t>> };
		for (unsigned int i = 0; i < samples_per_pixel; ++i) {
			vector<vector<T>> _d(rows, vector<T>(cols));
			for (unsigned int j = 0; j < rows; ++j) {
				for (unsigned int k = 0; k < cols; ++k) {
					_d[j][k] = (data[i * row * col + j * cols + k]);
				}
			}
			data.push_back(_d);
		}*/
		return true;
	}

	template<typename T>
	static bool confluency_trigger(const T* data, unsigned int cols, unsigned int rows, int samples_per_pixel) {
		LOGGER_INFO("confluency_trigger ita called.");
	
		return true;
	}

};

#endif
#endif