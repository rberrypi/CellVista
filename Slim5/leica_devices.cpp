#include "stdafx.h"
#if (HAS_LEICA ||(BUILD_ALL_DEVICES_TARGETS))
#include <reuse/proptools.h>
#include <boost/core/noncopyable.hpp>
#include "leica_devices.h"
#include <iostream>
#include <future>
#include <ahm.h>      // include AHM header
#include <ahwbasic.h> // include basic control interfaces (BasicControlValue, MetricsConverters, MetricsConverter)
#include <ahwprop2.h>
#include <ahwmicpropid.h>
#include <ahwmic.h>   // include microscope specific definitions
#include <reuse/ahmtools.h>
#include "qli_runtime_error.h"
#include <array>
#pragma comment(lib, "ahmcore.lib")
#include <sstream>
#define LEICA_ERROR() leica_display_error(__FILE__,__LINE__)
void leica_display_error(const char* file, const int line)
{
	std::stringstream ss;
	ss << "Leica Error: " << line << ":" << file;
	qli_runtime_error(ss.str());
}
//false positive on control paths
#define LEICATRY try{
#define LEICACATCH }catch (...){LEICA_ERROR();}

ahm::Unit* find_unit(ahm::Unit* p_unit, const ahm::TypeId type_id)
{
	// test unit's type for given type_id
	if (p_unit && p_unit->type()) {
		if (p_unit->type()->isA(type_id)) {
			return p_unit; // ok it is!
		}
		if (p_unit->units()) {
			// recursively find unit in child units
			for (auto i = 0; i < p_unit->units()->numUnits(); i++) {
				const auto p_deep_unit = find_unit(p_unit->units()->getUnit(i), type_id);
				if (p_deep_unit) {
					return p_deep_unit; // stop recursion
				}
			}
		}
	}
	return nullptr; // unit with type id was not found
}

ahm::Unit* find_unit(ahm::Unit* p_unit, const ahm::TypeId type_id, const std::string& human_label)
{
	//note human label is for fun?
	const auto test = find_unit(p_unit, type_id);
	if (test == nullptr)
	{
		const auto msg = std::string("Leica can't find a ") + human_label;
		qli_runtime_error(msg);
	}
	return test;
}
struct leica_named_property
{
	EProperties level_one;//PROP_PORTS
	EProperties level_two;//PROP_PORTS_PORTS
	EProperties level_three;//PROP_PORTS_PORT
	EProperties name_property;//PROP_PORTS_PORT_NAME
	leica_named_property() noexcept : leica_named_property(EProperties(), EProperties(), EProperties(), EProperties()) {}
	leica_named_property(const EProperties& level_one, const EProperties& level_two, const EProperties& level_three, const EProperties& name_property) noexcept:level_one(level_one), level_two(level_two), level_three(level_three), name_property(name_property) {}
	bool operator==(const leica_named_property& rhs) const
	{
		return (level_one == rhs.level_one)
			&& (level_two == rhs.level_two)
			&& (level_three == rhs.level_three)
			&& (name_property == rhs.name_property);
	}


};

struct leica_swichero : boost::noncopyable, leica_named_property
{
	ahm::BasicControlValue* p_basic_control_value;
	ahm::Properties* m_pProperties;
	struct switcher_names_and_indexes
	{
		int internal_index;//usually start from 1
		std::string name;
	};
	std::vector<switcher_names_and_indexes> switcher_labels;
	int min_v_c, max_v_c;
	mutable std::mutex move_lock;//can't find thread safety guarantees
	leica_swichero(ahm::Unit* p_default_unit, const ahm::TypeId& type_id, const std::string& human_label)
	{
		const auto p_unit = find_unit(p_default_unit, type_id, human_label);
		p_basic_control_value = find_itf<ahm::BasicControlValue>(p_unit, ahm::IID_BASIC_CONTROL_VALUE);
		m_pProperties = find_itf_version<ahm::Properties>(p_unit, ahm::Properties::IID, ahm::Properties::INTERFACE_VERSION);
		min_v_c = p_basic_control_value->minControlValue();
		max_v_c = p_basic_control_value->maxControlValue();
	}

	leica_swichero(ahm::Unit* p_default_unit, const ahm::TypeId& type_id, const std::string& human_label, const leica_named_property& named_property) : leica_swichero(p_default_unit, type_id, human_label)
	{
		static_cast<leica_named_property&>(*this) = named_property;
		if (!(named_property == leica_named_property()))
		{
			for (auto i = min_v_c; i <= max_v_c; ++i)// this list is inclusive fuck them
			{
				std::string value = "Can't Read";
				const auto success = get_property_name(value, i);
#if _DEBUG
				if (!success)
				{
					qli_runtime_error();
				}
#endif
				switcher_names_and_indexes item = { i,value };
				switcher_labels.emplace_back(item);
			}
		}
	}
	void move_to_idx(const int idx) const
	{
		LEICATRY
			const auto valid_idx = idx >= min_v_c && idx <= max_v_c;
#if _DEBUG
		{
			if (!valid_idx)
			{
				qli_runtime_error();
			}
		}
#endif
		if (valid_idx)
		{
			std::unique_lock<std::mutex> lk(move_lock);
			try
			{
				p_basic_control_value->setControlValue(idx);
			}
			catch (ahm::Exception&)
			{
				//happens when you set an empty channel, oh well, could be worse
			}
		}
		LEICACATCH
	}
	int get_idx() const
	{
		LEICATRY
			const auto value = p_basic_control_value->getControlValue();
		return value;
		LEICACATCH
			return 0;
	}
private:
	virtual ahm::Properties* getIndexStructPorts(iop::int32 index) {
		if (m_pProperties) {
			ahm::Property* pProp = m_pProperties->findProperty(level_one);
			const auto has_prop = ahm::property_tools::properties(pProp);
			if (has_prop) {
				pProp = ahm::property_tools::properties(pProp)->findProperty(level_two);
				ahm::PropertyValue* pElem = ahm::property_tools::getIndexedValue(pProp, index);
				if (ahm::property_tools::properties(pElem)) {
					pProp = ahm::property_tools::properties(pElem)->findProperty(level_three);
					return ahm::property_tools::properties(pProp);
				}
			}
		}
		return 0;
	}

	bool get_property_name(std::string& strbuf, const iop::int32 index)
	{
		std::string strname;
		const auto struct_port = this->getIndexStructPorts(index);
		if (ahm::property_tools::getStringValue(struct_port, name_property, strname)) {
			strbuf = strname;
			return true;
		}
		return false;
	}
};

struct leica_drive final : private leica_swichero
{
	ahm::MetricsConverter* p_microns_converter;
	explicit leica_drive(ahm::Unit* p_default_unit, const ahm::TypeId& type_id, const std::string& human_label) : leica_swichero(p_default_unit, type_id, human_label)
	{
		LEICATRY
			p_microns_converter = p_basic_control_value->metricsConverters()->findMetricsConverter(ahm::METRICS_MICRONS);
		LEICACATCH
	}
	double min_v() const
	{
		LEICATRY
			return p_microns_converter->getMetricsValue(min_v_c);
		LEICACATCH
			return qQNaN();
	}
	double max_v() const
	{
		LEICATRY
			return p_microns_converter->getMetricsValue(max_v_c);
		LEICACATCH
			return qQNaN();
	}
	void move_to_microns(const double microns) const
	{
		LEICATRY
			const auto z_unitless = p_microns_converter->getControlValue(microns);//some protection
		move_to_idx(z_unitless);
		LEICACATCH
	}
	float get_microns() const
	{
		LEICATRY
			const auto z_unitless = get_idx();
		return p_microns_converter->getMetricsValue(z_unitless);
		LEICACATCH
			return qQNaN();
	}
	//does this need a destructor?
};

struct leica_xy_drive final : private boost::noncopyable
{
	//ahm::Unit* pStageUnit;
	std::unique_ptr<leica_drive> x, y;
	explicit leica_xy_drive(ahm::Unit* p_default_unit) : x(
		std::make_unique<leica_drive>(p_default_unit, ahm::MICROSCOPE_X_UNIT, "Stage X")
	), y(
		std::make_unique<leica_drive>(p_default_unit, ahm::MICROSCOPE_Y_UNIT, "Stage Y")
	)
	{
	}
	void move_to(double position_x, double position_y, const bool async = false) const
	{
		const auto wait_x = std::async(std::launch::async, &leica_drive::move_to_microns, x.get(), position_x);
		const auto wait_y = std::async(std::launch::async, &leica_drive::move_to_microns, y.get(), position_y);
		if (!async)
		{
			wait_x.wait();
			wait_y.wait();
		}
	}
	std::pair<float, float> get() const
	{
		return std::make_pair(x->get_microns(), y->get_microns());
	}
	std::array<double, 4> get_stage_rect() const
	{
		const std::array<double, 4> msvc_is_for_noobs = { x->min_v(), y->min_v(), x->max_v(), y->max_v() };
		return msvc_is_for_noobs;
	}
};


struct leica_global_contrast final : private boost::noncopyable
{
	ahm::MicroscopeContrastingMethods* p_methods;
	mutable std::mutex move_lock;//can't find thread safety guarantees
	std::vector<std::pair<int, std::string>> options;
	explicit leica_global_contrast(ahm::Unit* p_microscope)
	{
		LEICATRY
			p_methods = find_itf<ahm::MicroscopeContrastingMethods>(p_microscope, ahm::IID_MICROSCOPE_CONTRASTING_METHODS);
		auto p_id_list = p_methods->supportedMethods();
		auto p_id_names = p_methods->methodNames();
		std::vector<std::pair<int, std::string>> options_temp;
		for (auto i = 0; i < p_id_list->numIds(); i++)
		{
			auto id = p_id_list->getId(i);
			const auto name = p_id_names->findName(id);
			options_temp.emplace_back(id, std::string(name));
		}
		//make the special SLIM channel, must be zero
		const auto find_n_make = [&](const char* label_in, const char* label_out, const bool required)
		{
			auto iterator = std::find_if(options_temp.begin(), options_temp.end(),
				[label_in](const std::pair<int, std::string>& chan) {return std::strcmp(chan.second.c_str(), label_in) == 0; });
			if (iterator == options_temp.end())
			{
				if (required)
				{
					const auto msg = std::string("Can't find a ") + std::string(label_in);
					qli_runtime_error(msg);
				}
			}
			else
			{
				options.emplace_back(iterator->first, label_out);
			}
		};
		try
		{
			find_n_make("TL-PH", "SLIM", true);
		}
		catch (std::runtime_error&)
		{
			find_n_make("TL-BF", "SLIM", true);
		}
		find_n_make("FLUO", "FLUO", false);
		LEICACATCH
	}
	void set_contrast_method(const int external_idx)
	{
		LEICATRY
			std::unique_lock<std::mutex> lk(move_lock);
		const auto internal_idx = options.at(external_idx);
		p_methods->setContrastingMethod(internal_idx.first);
		LEICACATCH
	}
};

struct leica_contrast_amalgamatrix final : private boost::noncopyable
{
	std::unique_ptr<leica_swichero> fl;
	std::unique_ptr<leica_swichero> tl;
	std::unique_ptr<leica_swichero> rl;
	std::unique_ptr<leica_swichero> side_ports;
	std::unique_ptr<leica_global_contrast> gc;
	struct rl_channel_settings
	{
		bool is_fl;
		int internal_fl_number;//usually starts at 1,2,3 inclusive
		bool uses_light;
		std::string name;
		rl_channel_settings(const bool is_fl, const int internal_fl_number, const bool uses_light, const std::string& name) : is_fl(is_fl), internal_fl_number(internal_fl_number), uses_light(uses_light), name(name) {}
	};
	std::vector<rl_channel_settings> internal_channels;
	struct internal_light_path
	{
		int internal_idx;//starts from 1 when valid
		std::string name;
	};
	std::vector<internal_light_path> internal_light_path;
	leica_contrast_amalgamatrix()
	{
		const auto p_default_unit = theHardwareModel()->getUnit("");
		const auto p_microscope = find_unit(p_default_unit, ahm::MICROSCOPE, "Microscope");
		const auto make_or_error = [&](const char* name, ahm::emictid item, std::unique_ptr<leica_swichero>& bind, const leica_named_property& property = leica_named_property())
		{
			try
			{
				bind = std::make_unique<leica_swichero>(p_microscope, item, name, property);
			}
			catch (...)
			{
				bind = nullptr;//redundant
			}
			std::cout << name << " " << (bind ? "" : "Not") << " Found" << std::endl;
		};
		{
			const leica_named_property microsocpe_ports = { EProperties::PROP_PORTS, EProperties::PROP_PORTS_PORTS,EProperties::PROP_PORTS_PORT, EProperties::PROP_PORTS_PORT_NAME };
			make_or_error("MICROSCOPE_PORTS", ahm::MICROSCOPE_PORTS, this->side_ports, microsocpe_ports);
		}
		//So this isn't correct
		//const leica_named_property microsocpe_il = { EProperties::PROP_ILTURRET, EProperties::PROP_ILTURRET_CUBES,EProperties::PROP_ILTURRET_CUBE, EProperties::PROP_ILTURRET_CUBE_NAME };
		make_or_error("MICROSCOPE_IL_TURRET", ahm::MICROSCOPE_IL_TURRET, this->fl);
		make_or_error("MICROSCOPE_TL_SHUTTER", ahm::MICROSCOPE_TL_SHUTTER, this->tl);
		make_or_error("MICROSCOPE_IL_SHUTTER", ahm::MICROSCOPE_IL_SHUTTER, this->rl);


		//		ahm::Unit *pPortsUnit = findUnit(p_default_unit, ahm::MICROSCOPE_PORTS);
		try
		{
			gc = std::make_unique<leica_global_contrast>(p_microscope);
			std::cout << "Found a Leica global contrast control" << std::endl;
		}
		catch (...)
		{
			gc = nullptr;
			std::cout << "Couldn't find a Leica global contrast control" << std::endl;
		}
		//
		const auto slim_cube = 1;//note this doesn't match the GUI channel
		internal_channels.emplace_back(rl_channel_settings(false, slim_cube, false, "OFF"));
		internal_channels.emplace_back(rl_channel_settings(false, slim_cube, true, "SLIM"));

		{
			for (auto idx = fl->min_v_c; idx < fl->max_v_c; idx++)
			{
				const auto name = std::string("FL#") + std::to_string(idx);
				internal_channels.emplace_back(rl_channel_settings(true, idx, true, name));
			}
		}

		//
		//
	}
	void toggle_lights(const bool enable) const
	{
		tl->move_to_idx(enable ? 1 : 0);
		rl->move_to_idx(enable ? 1 : 0);
	}
	/*
	int get_channel() const
	{
		// is this used?
		//Needs to be rewritten so it searches the list?
		if (tl && rl)
		{
			const auto tl_on = tl->get_idx() == 0;
			const auto rl_on = rl->get_idx() == 0;
			if (!tl_on && !rl_on == 0)
			{
				return 0;//off
			}
			if (tl_on && rl_on)
			{
				return 1;
			}
		}
		if (fl != nullptr)
		{
			const auto offset = 1;
			//Index 1 corresponds to channel #2
			return fl->get_idx()+ offset;//?
		}
		return 1;//always lock to slim
	}
	*/
	void set_channel(const int external_idx)
	{
		const auto& val = internal_channels.at(external_idx);
		const auto open = val.uses_light;
		//note, some of these can be done in parallel for improved performance
		if (open)
		{
			if (gc != nullptr)
			{
				const auto contrasting_idx = val.is_fl ? 1 : 0;
				gc->set_contrast_method(contrasting_idx);
			}
			if (fl != nullptr)
			{
				fl->move_to_idx(val.internal_fl_number);
			}
		}
		if (tl != nullptr)
		{
			tl->move_to_idx(open ? 1 : 0);
		}
		if (rl != nullptr)
		{
			rl->move_to_idx(open ? 1 : 0);
		}
	}

};

void microscope_z_drive_leica::move_to_z_internal(const float z)
{
	zd_->move_to_microns(z);
}

float microscope_z_drive_leica::get_position_z_internal()
{
	return static_cast<float>(zd_->get_microns());
}

scope_limit_z microscope_z_drive_leica::get_z_drive_limits_internal()
{
	return{ zd_->min_v(),zd_->max_v() };
}

microscope_z_drive_leica::microscope_z_drive_leica()
{
	const auto p_default_unit = theHardwareModel()->getUnit("");
	const auto p_microscope = find_unit(p_default_unit, ahm::MICROSCOPE, "Microscope");
	zd_ = std::make_unique<leica_drive>(p_microscope, ahm::MICROSCOPE_Z_UNIT, "Z Drive");
	common_post_constructor();
}

void microscope_z_drive_leica::print_settings(std::ostream&) noexcept
{
	//nothing yet
}

void microscope_xy_drive_leica::move_to_xy_internal(const scope_location_xy& xy)
{
	xyd_->move_to(xy.x, xy.y);
}

scope_location_xy microscope_xy_drive_leica::get_position_xy_internal()
{
	const auto pos = xyd_->get();
	return{ pos.first,pos.second };
}

scope_limit_xy microscope_xy_drive_leica::get_stage_xy_limits_internal()
{
	auto rect = xyd_->get_stage_rect();
	const QPointF top_left(rect[0], rect[3]);
	const QPointF bottom_right(rect[2], rect[1]);
	return scope_limit_xy(QRectF(top_left, bottom_right).normalized());
}

microscope_xy_drive_leica::microscope_xy_drive_leica()
{
	const auto p_default_unit = theHardwareModel()->getUnit("");
	const auto p_microscope = find_unit(p_default_unit, ahm::MICROSCOPE, "Microscope");
	xyd_ = std::make_unique<leica_xy_drive>(p_microscope);
	common_post_constructor();
}


void microscope_xy_drive_leica::print_settings(std::ostream&)
{
	//
}

void microscope_channel_drive_leica::move_to_channel_internal(const int channel_idx)
{
	cmd_->set_channel(channel_idx);
}

int microscope_channel_drive_leica::get_channel_internal()
{
	return current_light_path_.scope_channel;
}

void microscope_channel_drive_leica::move_condenser_internal(const condenser_position& position)
{

}

condenser_position microscope_channel_drive_leica::get_condenser_internal()
{
	return static_cast<condenser_position>(current_light_path_);
}

condenser_nac_limits microscope_channel_drive_leica::get_condenser_na_limit_internal()
{
	//maybe implement this?!
	return condenser_nac_limits(0.09, 0.75, true);
}

void microscope_channel_drive_leica::toggle_lights(const bool enable)
{
	cmd_->toggle_lights(enable);
}

microscope_channel_drive_leica::~microscope_channel_drive_leica() = default;

microscope_channel_drive_leica::microscope_channel_drive_leica()
{
	cmd_ = std::make_unique<leica_contrast_amalgamatrix>();
	for (size_t idx = 0; idx < cmd_->internal_channels.size(); idx++)
	{
		const auto item = cmd_->internal_channels.at(idx);
		auto name = item.name;
		if (idx == scope_channel_drive::off_channel_idx)
		{
			name = scope_channel_drive::channel_off_str;
		}
		if (idx == scope_channel_drive::phase_channel_idx)
		{
			name = scope_channel_drive::channel_phase_str;
		}
		channel_names.push_back(name);
	}
	current_light_path_.scope_channel = get_channel_internal();
	{
		const auto& side_ports = cmd_->side_ports;

		if (side_ports)
		{
			const auto& labels = side_ports->switcher_labels;
			const auto functor = [](const leica_swichero::switcher_names_and_indexes& index)
			{
				return index.name;
			};
			std::transform(labels.begin(), labels.end(), std::back_inserter(light_path_names), functor);
		}
		this->has_light_path = !light_path_names.empty();
	}
	common_post_constructor();
}

void microscope_channel_drive_leica::print_settings(std::ostream&) noexcept
{
	//
}

void microscope_channel_drive_leica::move_to_light_path_internal(const int channel_idx)
{
	const auto& side_port = cmd_->side_ports;
	if (side_port)
	{
		const auto internal_idx = side_port->switcher_labels.at(channel_idx).internal_index;
		side_port->move_to_idx(internal_idx);
	}
}

int microscope_channel_drive_leica::get_light_path_internal()
{
	const auto& side_port = cmd_->side_ports;
	if (side_port)
	{
		const auto offset = (1);
		return side_port->get_idx() - offset;
	}
	return 0;
}

#endif