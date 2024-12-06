#include "stdafx.h"
#if SLM_PRESENT_ARDUINOCOM == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "arduino_retarder.h"
#include <iostream>
#include <boost/format.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
#include "qli_runtime_error.h"

template <class Archive>
void serialize(Archive& archive, arduino_retarder_settings& cc)
{

	archive(
		cereal::make_nvp("coeff_b", cc.coeff_b),
		cereal::make_nvp("coeff_m", cc.coeff_m)
	);
}
arduino_retarder::arduino_retarder() : slm_device(1, 1, true), com_persistent_device("ArduinoRetarder", CBR_9600, com_number_unspecified, "\r\n", "")
{
	std::cout << "Waiting for Arduino Hand Shake" << std::endl;
	listen_for("Starting", false);
	std::ifstream configfile(extra_settings_filename());
	if (configfile.is_open())
	{
		try
		{
			cereal::JSONInputArchive archive(configfile);
			archive(static_cast<arduino_retarder_settings&>(*this));
		}
		catch (...)
		{

		}
	}
	else
	{
		std::cout << "Warning can't find auxillary configuration file:" << extra_settings_filename() << std::endl;
	}
	if (!are_retarder_settings_set())
	{
		std::cout << "Arduino M=? ";
		// ReSharper disable CppDeprecatedEntity
		scanf_s("%f", &coeff_m);
		std::cout << "Arduino B=? ";
		scanf_s("%f", &coeff_b);
	}
	// ReSharper restore CppDeprecatedEntity
	{
		std::cout << "Arduino Coefficients: M=" << coeff_m << ", B=" << coeff_b << std::endl;
		listen_for("M=", false);
		com_send(std::to_string(coeff_m));
		listen_for("ACK", false);
		//
		listen_for("B=", false);
		com_send(std::to_string(coeff_b));
		listen_for("ACK", false);
	}
}

arduino_retarder::~arduino_retarder()
{
	const auto setting_path = extra_settings_filename();
	std::ofstream os(setting_path);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(static_cast<arduino_retarder_settings&>(*this));
		std::cout << "Writing settings file to:" << setting_path << std::endl;
	}
	else
	{
		std::cout << "Warning can't write settings file to: " << setting_path << std::endl;
	}
}

void arduino_retarder::load_frame_internal(const int)
{
	qli_not_implemented();
}

void arduino_retarder::set_frame_internal(const int)
{
	qli_not_implemented();
	/*
	std::stringstream ss;
	auto expected_value = framedata_[num].frame.front();
	const auto interal_pattern_number = num % interal_patterns_.size();
	const auto existing_value = interal_patterns_[interal_pattern_number];
	if (expected_value!= existing_value)
	{
		const auto modulo_pattern = num % internal_stability_.size();
		const auto stability = internal_stability_[modulo_pattern];
		load_frame_internal(num, &expected_value, stability);
	}
	ss << boost::format("F%d") % interal_pattern_number;
	const auto string_to_send = ss.str();
	//std::cout << string_to_send << std::endl;
	com_send(string_to_send);
	*/
}

void arduino_retarder::hardware_trigger_sequence_internal(const size_t, const channel_settings&)
{
	qli_not_implemented();
}

#endif