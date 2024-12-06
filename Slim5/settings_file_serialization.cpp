#include "stdafx.h"
#include "settings_file.h"
#include <cereal/archives/json.hpp>

#include "fixed_hardware_settings_cerealization.h"
#include <fstream>
#include "qli_runtime_error.h"
template <class Archive>
void serialize(Archive& archive, slm_pattern_generation& cc)
{

	archive(
		cereal::make_nvp("modulator_mode", cc.modulator_mode)
		, cereal::make_nvp("darkfield", cc.darkfield)
		, cereal::make_nvp("darkfield_samples", cc.darkfield_samples)
	);

}

template <class Archive>
void serialize(Archive& archive, settings_file& cc)
{

	archive(
		cereal::make_nvp("fixed_hardware_settings", cereal::base_class<fixed_hardware_settings>(&cc))
		, cereal::make_nvp("slm_pattern_generation", cereal::base_class<slm_pattern_generation>(&cc))
	);

}

[[nodiscard]] settings_file settings_file::read(const std::string& filename, bool& okay)
{
	settings_file return_this;
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
	return_this.file_path = filename;
	return return_this;
}

[[nodiscard]] bool settings_file::write() const
{
#if _DEBUG
	{
		if (!is_valid())
		{
			qli_invalid_arguments();
		}
	}
#endif
	std::ofstream os(file_path);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
	}
	else
	{
		const auto str = "Warning can't write " + file_path;
		std::cout << str << std::endl;
#if _DEBUG
		qli_runtime_error(str);
#endif
		return false;
	}
#if _DEBUG
	{
		auto okay=false;
		const auto settings_file = settings_file::read(file_path,okay);
		const auto mismatch = !this->item_approx_equals(settings_file);
		if (!okay || mismatch)
		{
			qli_runtime_error();
		}
	}
#endif
	return true;
}