#include "stdafx.h"
#include "emphemeral_main_gui_settings.h"
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
template <class Archive>
void serialize(Archive& archive, ephemeral_settings& cc)
{
	archive(
		cereal::make_nvp("slm_text", cc.slm_text),
		cereal::make_nvp("last_directory_text", cc.last_directory_text),
		cereal::make_nvp("last_channel", cc.last_channel)
	);
}

void ephemeral_settings::write(const std::string& filename)
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

ephemeral_settings::ephemeral_settings(const std::string& filename_to_read) : last_channel(0)
{
	std::ifstream config_file(filename_to_read);
	cereal::JSONInputArchive archive(config_file);
	archive(*this);
}



