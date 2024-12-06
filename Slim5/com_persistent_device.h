#pragma once
#ifndef COM_PERSISTENT_DEVICE_H
#define COM_PERSISTENT_DEVICE_H
#include <mutex>
#include <boost/core/noncopyable.hpp>
#include <queue>

class com_persistent_device : boost::noncopyable
{
	HANDLE port_handle_;
	DWORD errors_;
	COMSTAT status_;
	std::recursive_mutex com_lock_;//careful
	std::queue<std::string> msgs_;
	void check_com_is_being_serviced() const;
	std::string terminator_;
	std::string full_path_to_arduino_program_;
	std::string device_name_;
	int preferred_baud_rate_;
	void wait_for_messages();
	static bool program_arduino(const std::string& asset_name, int com_port);
	std::string read_file_buffer_;//use a more legitimate buffer?
public:
	int com_port;
	virtual ~com_persistent_device();
	explicit com_persistent_device(std::string  device_name, int preferred_baud_rate, int com_port_number, std::string  terminator, const std::string& full_path_to_arduino_binary="");
	[[nodiscard]] int prompt_for_com() const;
	[[nodiscard]] std::string filepath() const;
	[[nodiscard]] bool has_messages() const noexcept
	{
		return !msgs_.empty();
	}
	void switch_baud_rate(int baud_rate) const;
	void listen_for(const std::string& listen_for, bool clear_top = true);
	void com_send(const std::string& message);
	[[nodiscard]] std::string pop_com_message();//waits for a new line
	constexpr static auto com_number_unspecified = -1;//CBR_115200
	constexpr static auto default_buad_rate = -1;//CBR_115200
};

#define THROW_COM_ERROR() throw_com_error_internal(__FILE__, __LINE__ )
void throw_com_error_internal(const char* file, int line);
#endif