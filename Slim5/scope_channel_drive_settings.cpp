#include "stdafx.h"
#include "scope.h"
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"
// ReSharper disable once CppUnusedIncludeDirective
#include <cereal/types/chrono.hpp>
#include <fstream>

const std::string scope_channel_drive_settings::settings_name = "scope_settings.json";

template <class Archive>
void serialize(Archive& archive, scope_channel_drive_settings& cc)
{

	archive(
		cereal::make_nvp("phase_channel_alias", cc.phase_channel_alias),
		cereal::make_nvp("is_transmission", cc.is_transmission),
		cereal::make_nvp("channel_off_threshold", cc.channel_off_threshold)
	);
}

const static std::string scope_setting_error_str = "Warning couldn't get scope setting file at:";

void scope_channel_drive_settings::load_settings()
{
	try
	{
		std::ifstream configuration_file(settings_name);
		const auto success = configuration_file.is_open();
		if (success)
		{
			cereal::JSONInputArchive archive(configuration_file);
			archive(*this);
		}
	}
	catch (...)
	{
		std::cout << scope_setting_error_str << settings_name << std::endl;
	}
}

void scope_channel_drive_settings::save_settings()
{
	try
	{
		std::ofstream os(settings_name);
		const auto success = os.is_open();
		if (success)
		{
			cereal::JSONOutputArchive archive(os);
			archive(*this);
		}
	}
	catch (...)
	{ //-V565
		std::cout << scope_setting_error_str << settings_name;
	}
}