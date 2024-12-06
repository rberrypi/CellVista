#include "stdafx.h"
#include <cereal/archives/json.hpp>
#include <fstream>
#include "channel_settings.h"
#include "fixed_hardware_settings_cerealization.h"
#include "live_gui_settings_cerealization.h"
#include "per_modulator_saveable_settings_cerealization.h"

bool live_gui_settings::write(const std::string& filename) const
{
	std::ofstream os(filename);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
		return true;
	}
	std::cout << "Warning can't write " << filename << std::endl;
	return false;
}


template <class Archive>
void serialize(Archive& archive, channel_settings& cc)
{

	archive(
		cereal::make_nvp("fixed_hardware_settings", cereal::base_class<fixed_hardware_settings>(&cc))
		, cereal::make_nvp("live_gui_settings", cereal::base_class<live_gui_settings>(&cc))
		, cereal::make_nvp("modulator_settings", cc.modulator_settings)
	);

}

live_gui_settings live_gui_settings::read(const std::string& filename, bool& okay)
{
	live_gui_settings return_this;
	okay = false;
	try
	{
		std::ifstream os(filename);
		if (os.is_open())
		{
			cereal::JSONInputArchive archive(os);
			archive(return_this);
			okay = true;
		}
	}
	catch (...)
	{

	}
	return return_this;
}


void channel_settings::write(const std::string& filename)
{
	std::ofstream os(filename);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
	}
	else
	{
		std::cout << "Warning can't write " << filename << std::endl;
	}
}
