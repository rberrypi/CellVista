#include "stdafx.h"
#include "slm_pattern_model.h"
#include "qli_runtime_error.h"
#include <iostream>
#include <set>

#define distorted_donut_x_center (0)
#define distorted_donut_y_center (1)
#define distorted_donut_inner_diameter (2)
#define distorted_donut_outer_diameter (3)
#define distorted_donut_ellipticity_e (4)
#define distorted_donut_ellipticity_f (5)
#define phase_shift_pattern_pattern_mode (6)
#define phase_shift_pattern_filepath (7)
#define phase_shift_pattern_slm_value (8)
#define phase_shift_pattern_slm_background (9)
#define psi_function_pairs_r_t (10)
#define psi_function_pairs_r_b (11)
#define psi_function_pairs_r_c (12)
#define psi_function_pairs_g_t (13)
#define psi_function_pairs_g_b (14)
#define psi_function_pairs_g_c (15)
#define psi_function_pairs_b_t (16)
#define psi_function_pairs_b_b (17)
#define psi_function_pairs_b_c (18)

struct psi_setting_labels
{
	QString label;
	bool requires_color;
	typedef std::set<slm_mode> psi_settings_modes;
	psi_settings_modes supported_modes;
	[[nodiscard]] bool is_visible(const slm_mode mode, const bool is_color) const
	{
		const auto has_mode = supported_modes.find(mode) != supported_modes.end();
		//const auto has_mode = supported_modes.contains(mode);
		const auto color_show = requires_color ? is_color : true;
		return has_mode && color_show;
	}
	psi_setting_labels(const QString& label, const bool requires_color, const psi_settings_modes& psi_settings_modes) : label(label), requires_color(requires_color), supported_modes(psi_settings_modes)
	{

	}
	typedef std::map<int, const psi_setting_labels> psi_setting_map;
	const static psi_setting_map settings;
};

const auto all_modes = { slm_mode::single_shot,slm_mode::two_shot_lcvr,slm_mode::slim,slm_mode::qdic,slm_mode::darkfield,slm_mode::custom_patterns };
const auto qdic_like = { slm_mode::single_shot,slm_mode::two_shot_lcvr,slm_mode::qdic,slm_mode::slim };
const auto dot_like = { slm_mode::slim,slm_mode::darkfield };

const psi_setting_labels::psi_setting_map psi_setting_labels::settings = {
	{distorted_donut_x_center,psi_setting_labels("X",false,dot_like)},
	{distorted_donut_y_center,psi_setting_labels("Y",false,dot_like)},
	{distorted_donut_inner_diameter,psi_setting_labels("ID",false,dot_like)},
	{distorted_donut_outer_diameter,psi_setting_labels("OD",false,dot_like)},
	{distorted_donut_ellipticity_e,psi_setting_labels("e",false,dot_like)},
	{distorted_donut_ellipticity_f,psi_setting_labels("f",false,dot_like)},
	{phase_shift_pattern_pattern_mode,psi_setting_labels("M",false,all_modes)},
	{phase_shift_pattern_filepath,psi_setting_labels("Path",false,{slm_mode::custom_patterns})},
	{phase_shift_pattern_slm_value,psi_setting_labels("V",false,all_modes)},
	{phase_shift_pattern_slm_background,psi_setting_labels("B",false,all_modes)},

	{psi_function_pairs_r_t,psi_setting_labels("RT",false,all_modes)},
	{psi_function_pairs_r_b,psi_setting_labels("RB",false,all_modes)},
	{psi_function_pairs_r_c,psi_setting_labels("RC",false,all_modes)},

	{psi_function_pairs_g_t,psi_setting_labels("GT",true,all_modes)},
	{psi_function_pairs_g_b,psi_setting_labels("GB",true,all_modes)},
	{psi_function_pairs_g_c,psi_setting_labels("GC",true,all_modes)},

	{psi_function_pairs_b_t,psi_setting_labels("BT",true,all_modes)},
	{psi_function_pairs_b_b,psi_setting_labels("BB",true,all_modes)},
	{psi_function_pairs_b_c,psi_setting_labels("BC",true,all_modes)},
};

bool per_pattern_modulator_settings_patterns_model::is_visible(const int idx, const  slm_mode mode, const bool is_color)
{
	return psi_setting_labels::settings.at(idx).is_visible(mode, is_color);
}

QVariant per_pattern_modulator_settings_patterns_model::data(const QModelIndex& index, const int role) const
{
	if (index.isValid() && (role == Qt::DisplayRole || role == Qt::EditRole))
	{
		const auto& item = patterns.at(index.row());
		switch (index.column())
		{
		case distorted_donut_x_center:
			return item.x_center;
		case distorted_donut_y_center:
			return item.y_center;
		case distorted_donut_inner_diameter:
			return item.inner_diameter;
		case distorted_donut_outer_diameter:
			return item.outer_diameter;
		case distorted_donut_ellipticity_e:
			return item.ellipticity_e;
		case distorted_donut_ellipticity_f:
			return item.ellipticity_f;
		case phase_shift_pattern_pattern_mode:
			return static_cast<uint>(item.pattern_mode);
		case phase_shift_pattern_filepath:
			return QString::fromStdString(item.filepath);
		case phase_shift_pattern_slm_value:
			return item.slm_value;
		case phase_shift_pattern_slm_background:
			return item.slm_background;

		case psi_function_pairs_r_t:
			return item.weights.at(0).top;
		case psi_function_pairs_r_b:
			return item.weights.at(0).bot;
		case psi_function_pairs_r_c:
			return item.weights.at(0).constant;

		case psi_function_pairs_g_t:
			if (item.weights.size()<2)
			{
				return QVariant();
			}			
			return item.weights.at(1).top;
		case psi_function_pairs_g_b:
			if (item.weights.size()<2)
			{
				return QVariant();
			}				
			return item.weights.at(1).bot;
		case psi_function_pairs_g_c:
			if (item.weights.size()<2)
			{
				return QVariant();
			}				
			return item.weights.at(1).constant;

		case psi_function_pairs_b_t:
			if (item.weights.size()<3)
			{
				return QVariant();
			}				
			return item.weights.at(2).top;
		case psi_function_pairs_b_b:
			if (item.weights.size()<3)
			{
				return QVariant();
			}					
			return item.weights.at(2).bot;
		case psi_function_pairs_b_c:
			if (item.weights.size()<3)
			{
				return QVariant();
			}					
			return item.weights.at(2).constant;

		default:
			qli_not_implemented();
		}
	}
	if (role == Qt::TextAlignmentRole)
	{
		return Qt::AlignCenter;
	}
	return QVariant();
}

bool per_pattern_modulator_settings_patterns_model::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if ((role == Qt::EditRole || role == Qt::UserRole) && index.isValid())
	{
		auto& item = patterns.at(index.row());
		switch (index.column())
		{
		case distorted_donut_x_center:
			item.x_center = value.toFloat();
			break;
		case distorted_donut_y_center:
			item.y_center = value.toFloat();
			break;
		case distorted_donut_inner_diameter:
			item.inner_diameter = value.toFloat();
			break;
		case distorted_donut_outer_diameter:
			item.outer_diameter = value.toFloat();
			break;
		case distorted_donut_ellipticity_e:
			item.ellipticity_e = value.toFloat();
			break;
		case distorted_donut_ellipticity_f:
			item.ellipticity_f = value.toFloat();
			break;
		case phase_shift_pattern_pattern_mode:
		{
			const auto pattern_idx = value.toUInt();
			item.pattern_mode = static_cast<slm_pattern_mode>(qBound(0u, pattern_idx, static_cast<uint>(slm_pattern_mode::count)));
			break;
		}
		case phase_shift_pattern_filepath:
			item.filepath = value.toString().toStdString();
			break;
		case phase_shift_pattern_slm_value:
			item.slm_value = qBound(0.f, value.toFloat(), 255.f);
			break;
		case phase_shift_pattern_slm_background:
			item.slm_background = qBound(0.f, value.toFloat(), 255.f);
			break;
		case psi_function_pairs_r_t:
			item.weights.at(0).top = value.toFloat();
			break;
		case psi_function_pairs_r_b:
			item.weights.at(0).bot = value.toFloat();
			break;
		case psi_function_pairs_r_c:
			item.weights.at(0).constant = value.toFloat();
			break;

		case psi_function_pairs_g_t:
			if (item.weights.size()<2)
			{
				return false;
			}
			item.weights.at(1).top = value.toFloat();
			break;
		case psi_function_pairs_g_b:
			if (item.weights.size()<2)
			{
				return false;
			}			
			item.weights.at(1).bot = value.toFloat();
			break;
		case psi_function_pairs_g_c:
			if (item.weights.size()<2)
			{
				return false;
			}			
			item.weights.at(1).constant = value.toFloat();
			break;

		case psi_function_pairs_b_t:
			if (item.weights.size()<3)
			{
				return false;
			}			
			item.weights.at(2).top = value.toFloat();
			break;
		case psi_function_pairs_b_b:
			if (item.weights.size()<3)
			{
				return false;
			}	
			item.weights.at(2).bot = value.toFloat();
			break;
		case psi_function_pairs_b_c:
			if (item.weights.size()<3)
			{
				return false;
			}				
			item.weights.at(2).constant = value.toFloat();
			break;

		default:
			qli_not_implemented();
		}
		emit dataChanged(index, index);//not sure if correct
		return true;
	}
	return false;
}


per_pattern_modulator_settings_patterns_model::per_pattern_modulator_settings_patterns_model(QObject* parent) : QAbstractTableModel(parent)
{
}

int per_pattern_modulator_settings_patterns_model::rowCount(const QModelIndex& parent) const
{
	return patterns.size();
}

int per_pattern_modulator_settings_patterns_model::columnCount(const QModelIndex&) const
{
	return psi_function_pairs_b_c + 1;
}

//for editing

//for resizing
bool per_pattern_modulator_settings_patterns_model::insertRows(int position, int rows, const QModelIndex& index) {
	Q_UNUSED(index);
	beginInsertRows(QModelIndex(), position, position + rows - 1);
	for (auto idx = position; idx < position + rows; ++idx)
	{
		per_pattern_modulator_settings blank_pattern;
		patterns.insert(patterns.begin() + position, blank_pattern);
	}
	emit endInsertRows();
	return true;
}

bool per_pattern_modulator_settings_patterns_model::removeRows(int position, int rows, const QModelIndex& index)
{
	//From position remove rows number
	const auto first = position - rows + 1;
	const auto last = position;
	beginRemoveRows(QModelIndex(), first, last);
	patterns.erase(patterns.begin() + first, patterns.begin() + last);
	endRemoveRows();
	return true;
}

bool per_pattern_modulator_settings_patterns_model::insertColumns(int column, int count, const QModelIndex& parent) {
	return false;
}

QVariant per_pattern_modulator_settings_patterns_model::headerData(int section, Qt::Orientation orientation, int role) const {
	if (role != Qt::DisplayRole)
	{
		return QVariant();
	}
	if (orientation == Qt::Horizontal)
	{
		return psi_setting_labels::settings.at(section).label;
	}
	if (orientation == Qt::Vertical)
	{
		return QString::number(section);
	}
	return QVariant();
}

Qt::ItemFlags per_pattern_modulator_settings_patterns_model::flags(const QModelIndex& /*index*/) const
{
	//actually special item shouldn't be selectable ?
	return Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled;
}

void per_pattern_modulator_settings_patterns_model::set_per_pattern_modulator_settings_patterns(const per_pattern_modulator_settings_patterns& patterns)
{
#if _DEBUG
	{
		const auto functor = [](const per_pattern_modulator_settings& pattern)
		{
			return pattern.is_valid();
		};
		const auto all_settings_valid = !patterns.empty() && std::all_of(patterns.begin(), patterns.end(), functor);
		if (!all_settings_valid)
		{
			qli_invalid_arguments();
		}
	}
#endif
	const int new_size = patterns.size();
	const auto change = new_size - rowCount();
	if (change > 0)
	{
		insertRows(rowCount(), change);
	}
	if (change < 0)
	{
		removeRows(rowCount() - 1, abs(change));
	}
	this->patterns = patterns;
#if _DEBUG
	{

		const auto what_we_got = get_pattern_modulator_settings_patterns();
		if (!per_modulator_saveable_settings::per_pattern_modulator_settings_patterns_approx_equals(what_we_got, patterns))
		{
			qli_runtime_error();
		}
	}
#endif
	const auto top_left = createIndex(0, 0);
	const auto bottom_right = createIndex(rowCount(), columnCount());
	emit dataChanged(top_left, bottom_right);//not sure if correct
}

const per_pattern_modulator_settings_patterns& per_pattern_modulator_settings_patterns_model::get_pattern_modulator_settings_patterns() const
{
	return patterns;
}
