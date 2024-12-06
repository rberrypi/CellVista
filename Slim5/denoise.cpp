#include "stdafx.h"
#include "phase_processing.h"
int denoise_setting::max_denoise_patterns()
{
	const auto functor = [](const std::pair<denoise_mode, denoise_setting>& p1, const  std::pair<denoise_mode, denoise_setting>& p2) {
		return p1.second.patterns < p2.second.patterns; };
	static auto x = std::max_element(settings.begin(), settings.end(), functor)->second.patterns;
	return x;
}

int denoise_setting::max_denoise_setting_characters()
{
	const auto functor = [](const std::pair<denoise_mode, denoise_setting>& p1, const  std::pair<denoise_mode, denoise_setting>& p2) {
		return p1.second.label.size() < p2.second.label.size(); };
	static auto x = std::max_element(settings.begin(), settings.end(), functor)->second.label.size();
	return x;
}

const denoise_setting::denoise_settings_map denoise_setting::settings =
{
	{ denoise_mode::off,{ "Off",1 } },
	{ denoise_mode::average,{ "Average", 5 } },
	{ denoise_mode::median,{ "Median", 5 } },
	{ denoise_mode::hybrid,{ "Hybrid" ,10 } }
};