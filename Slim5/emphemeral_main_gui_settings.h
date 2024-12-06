#pragma once
#ifndef EPHEMERAL_SETTINGS_H
#define EPHEMERAL_SETTINGS_H
#include <string>
struct ephemeral_settings
{
	std::string slm_text, last_directory_text;
	int last_channel;
	explicit ephemeral_settings(const std::string& filename_to_read);
	void write(const std::string& filename);
	ephemeral_settings() noexcept:ephemeral_settings("", "", 0) {}
	ephemeral_settings(const std::string& slm_text, const std::string& last_directory_text, const int last_channel) : slm_text(slm_text), last_directory_text(last_directory_text), last_channel(last_channel) {}
};
#endif