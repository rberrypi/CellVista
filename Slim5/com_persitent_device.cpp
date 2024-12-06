#include "stdafx.h"
#include "com_persistent_device.h"
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
#include <utility>

#include "qli_runtime_error.h"

#define DEBUGGLE_RS_TWO_THIRTY_TWO 0

template <class Archive>
void serialize(Archive& archive, com_persistent_device& cc)
{
	archive(
		cereal::make_nvp("com_number", cc.com_port)
	);
}

void throw_com_error_internal(const char* file, const int line)
{
	const auto static_buffer_length = 256;
	char error_buff[static_buffer_length] = { 0 };
	FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(),
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), error_buff, static_buffer_length, nullptr);
	std::cout << "COM Error :" << file << ":" << line << " " << error_buff << std::endl;
	//pass message into runtime helper
	qli_runtime_error("COM Error");
}

com_persistent_device::com_persistent_device(std::string  device_name, const int preferred_baud_rate, const int com_port_number, std::string  terminator, const std::string& full_path_to_arduino_binary)
	: errors_(0), status_{ 0 }, terminator_(std::move(terminator)), device_name_(std::move(device_name)), preferred_baud_rate_(0), com_port(com_port_number)
{
	std::ifstream configuration_file(filepath());
	if (configuration_file.is_open())
	{
		cereal::JSONInputArchive archive(configuration_file);
		archive(*this);
	}
	else
	{
		std::cout << "Warning can't find device configuration file:" << filepath() << std::endl;
	}
	if (com_port == com_number_unspecified)
	{
		com_port = prompt_for_com();
	}
	//Okay now lets connect
	auto connection_success = false;
	const auto connection_attempts_max = 10;
	auto connection_attempt = 0;
	do
	{
		const auto program_success = program_arduino(full_path_to_arduino_binary, com_port);
		if (program_success)
		{
			auto port_string = R"(\\.\COM)" + std::to_string(com_port);
			port_handle_ = CreateFileA(port_string.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (port_handle_ != INVALID_HANDLE_VALUE)
			{
				if (preferred_baud_rate > 0)
				{
					switch_baud_rate(preferred_baud_rate);
				}
				{
					DCB dcb;
					const auto f_success3 = GetCommState(port_handle_, &dcb);
					std::cout << "COM" << com_port << " baud rate set to " << dcb.BaudRate << " check that this matches the DIP switch on the controller (if applicable)" << std::endl;
					if (!f_success3)
					{
						THROW_COM_ERROR();
					}
					connection_success = true;
				}
			}
		}
	} while (!connection_success && connection_attempt++ < connection_attempts_max);
	if (!connection_success)
	{
		THROW_COM_ERROR();
	}
}

com_persistent_device::~com_persistent_device()
{
	CloseHandle(port_handle_);
	//
	const auto setting_path = filepath();
	std::ofstream os(setting_path);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
		std::cout << "Writing settings file to:" << setting_path << std::endl;
	}
	else
	{
		std::cout << "Warning can't write settings file to: " << setting_path << std::endl;
	}
}

int com_persistent_device::prompt_for_com() const
{
	std::cout << "Could not connect to COM port assigned to " << device_name_ << " (currently " << std::to_string(com_port) << "), enter a new value:\n" << std::endl;
	int integer;
	while (!(std::cin >> integer))
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cout << "Invalid input.  Try again: ";
	}
	return integer;
}

std::string com_persistent_device::filepath() const
{
	return  device_name_ + std::string(".json");
}

void com_persistent_device::switch_baud_rate(const int baud_rate) const
{
	DCB dcb;
	SecureZeroMemory(&dcb, sizeof(DCB));
	GetCommState(port_handle_, &dcb);
	dcb.BaudRate = baud_rate;
	SetCommState(port_handle_, &dcb);
	GetCommState(port_handle_, &dcb);
	const auto cast_baud_rate = static_cast<int>(baud_rate);
	if (cast_baud_rate != baud_rate)
	{
		qli_runtime_error("Failed to Modify Baud Rate");
	}
}

void com_persistent_device::check_com_is_being_serviced() const
{
	if (msgs_.size() > 10)
	{
		std::cout << "COM messages keep growing without being serviced, " << msgs_.size() << " this is most likely a bug" << std::endl;
	}
}

void com_persistent_device::wait_for_messages()
{
	check_com_is_being_serviced();
	std::unique_lock<std::recursive_mutex> lk(com_lock_);
	DWORD errors;
	COMSTAT status;
	while (msgs_.empty())
	{
		const auto success = ClearCommError(port_handle_, &errors, &status);
		if (!success)
		{
			THROW_COM_ERROR();
		}
		if (status.cbInQue > 0)
		{
			DWORD bytes_read;
			const auto null_terminator_size = 1;
			std::vector<char> temp_buffer(status.cbInQue + null_terminator_size, 0);
			const auto success_if_non_zero = ReadFile(port_handle_, temp_buffer.data(), status.cbInQue, &bytes_read, nullptr);
			if (success_if_non_zero)
			{

				read_file_buffer_ = read_file_buffer_ + temp_buffer.data();
#if DEBUGGLE_RS_TWO_THIRTY_TWO==1
				std::cout << "Buffer " << temp_buffer.data() << "[" << temp_buffer.size() << "]" << std::endl;
#endif
				size_t position;
				while ((position = read_file_buffer_.find(terminator_)) != std::string::npos)
				{
					const auto token = read_file_buffer_.substr(0, position);
					read_file_buffer_.erase(0, position + terminator_.length());
					msgs_.push(token);
#if DEBUGGLE_RS_TWO_THIRTY_TWO==1
					std::cout << "MSGS " << token << "[" << msgs_.size() << "]" << std::endl;
					std::cout << "RESIDUE " << read_file_buffer << "[" << read_file_buffer.size() << "]" << std::endl;
#endif
				}
			}
		}
		windows_sleep(ms_to_chrono(1));
	}
}

void com_persistent_device::com_send(const std::string& message)
{
	std::unique_lock<std::recursive_mutex> lk(com_lock_);
	const auto to_send = message + terminator_;
	DWORD bytes_send;
#if DEBUGGLE_RS_TWO_THIRTY_TWO== 1
	std::cout << "Sending " << message << std::endl;
#endif
	const auto* as_c_str = reinterpret_cast<const void*>(to_send.c_str());
	const auto write_status = WriteFile(port_handle_, as_c_str, to_send.size(), &bytes_send, nullptr);
	if (!write_status)
	{
		THROW_COM_ERROR();
	}
	if (bytes_send != to_send.size())
	{
		THROW_COM_ERROR();
	}
}

void com_persistent_device::listen_for(const std::string& listen_for, const bool clear_top)
{
	//clear_top is for backwards compatibility
	while (true)
	{
		std::unique_lock<std::recursive_mutex> lk(com_lock_);
		while (!msgs_.empty())
		{
			const auto msg = msgs_.front();
#if DEBUGGLE_RS_TWO_THIRTY_TWO== 1
			std::cout << "Listening for " << listen_for << " got " << msg << std::endl;
#endif
			msgs_.pop();
			if (msg == listen_for)
			{
				if (clear_top)
				{
					msgs_ = std::queue<std::string>();
				}
				return;
			}
		}
		wait_for_messages();
	}
}

std::string com_persistent_device::pop_com_message()
{
	std::unique_lock<std::recursive_mutex> lk(com_lock_);
	while (true)
	{
		if (!msgs_.empty())
		{
			auto msg = msgs_.front();
			msgs_.pop();
#if DEBUGGLE_RS_TWO_THIRTY_TWO==1
			std::cout << "Popped " << msg << "[" << msgs_.size() << "]" << std::endl;
#endif
			return msg;
		}
		wait_for_messages();
	}
}
