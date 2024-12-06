#pragma once
#ifndef DVILSLM_H
#define DVILSLM_H
#include <Windows.h>
#include <mutex>
#include <thread>
#include "slm_device.h"
class lazy_wait;
struct dvi_slm_device_saveable_settings
{
	int preferred_monitor;
	dvi_slm_device_saveable_settings() : preferred_monitor(-1) {}
};
class dvi_slm_device  final : public slm_device, public dvi_slm_device_saveable_settings
{
public:
	explicit dvi_slm_device();
	virtual ~dvi_slm_device();
	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override;

protected:
	void load_frame_internal(int num)  override;
	void set_frame_internal(int frame_number) override;
	//
private:
	static int prompt_for_screen();
	std::mutex m_;
	std::condition_variable cv_;
	bool ping_, pong_;
	void window_loop(lazy_wait* lzw);
	static int window_event_loop();
	DWORD m_id_thread_;
	static	LRESULT CALLBACK window_proc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);//There can only be one!
	WNDCLASSEX wcl_;
	HWND      g_h_wnd_;
	HINSTANCE g_h_instance_;
	HDC       g_h_dc_;
	HGLRC     g_h_rc_;
	LONG g_window_width_, g_window_height_;
	//RECT rc_;
	void gl_loop(lazy_wait* lzw);
	bool gl_done_;
	unsigned int texture_;
	unsigned int  program_handle_;
	unsigned int  vao_handle_;
	std::vector<unsigned int> vbo_handles_;
	//
	const static unsigned int p = 1;
	unsigned int frame_;
	std::thread* window_loop_, * gl_loop_;
};
#endif