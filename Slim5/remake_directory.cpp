#include "stdafx.h"
//alternatively use qt?
//we still have to use boost for cgal
#include <iostream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
void remove_directory(const std::string& dir_path)
{
	const std::experimental::filesystem::path dir(dir_path);
	if (std::experimental::filesystem::exists(dir_path))
	{
		remove_all(dir);
	}
}

void remake_directory(const std::string& dir_path)
{
	const std::experimental::filesystem::path dir(dir_path);
	if (std::experimental::filesystem::exists(dir_path))
	{
		remove_all(dir);
	}
	const auto success = create_directories(dir);
	if (!success)
	{
		std::cout << "Failed to make " << dir_path << std::endl;
	}
}