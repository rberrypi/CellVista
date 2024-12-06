//#include "windows.h"
#include "stdafx.h"
#if SLM_PRESENT_MONITOR == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#pragma comment(lib, "glew32.lib")
#pragma comment(lib,"Opengl32.lib")
#include <GL/glew.h>
#include <GL/wglew.h>
#include <gl/GL.h>
#include <iostream>
#include <thread>
#include "dvi_slm_device.h"
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
#include <sstream>
#include "qli_runtime_error.h"
template <class Archive>
void serialize(Archive& archive, dvi_slm_device_saveable_settings& cc)
{
	archive(
		cereal::make_nvp("com_persistent_device", cc.preferred_monitor)
	);
}


class lazy_wait final : boost::noncopyable
{
	std::mutex m_;
	std::condition_variable cv_;
	bool r_;
public:
	lazy_wait() :r_(false) {}
	lazy_wait(lazy_wait const&) = delete;
	lazy_wait& operator=(lazy_wait const&) = delete;
	~lazy_wait()
	{
		wait_for_me();
	}
	void wait_for_me()
	{
		if (!r_)
		{//or if lock acquired?
			std::unique_lock<std::mutex> lk(m_);
			cv_.wait(lk, [&] {return r_; });
		}
	}
	void ready()
	{
		r_ = true;
		cv_.notify_one();
	}
};

#define OGL_ERROR_CHECK() ogl_error_check(__FILE__,__LINE__)
void ogl_error_check(const char* file, const int line)
{
#ifndef brave
	const auto err = glGetError();
	if (!err)
	{
		return;
	}
	const auto  err_string = reinterpret_cast<const char*>(gluErrorString(err));
	std::stringstream ss;
	
	ss << "Glu Error:" << file << ":" << line << ": " << err_string << std::endl;
	qli_runtime_error(ss.str());
#endif
}

#define OGL_SHADER_CHECK(sh) ogl_shader_check(sh,__FILE__,__LINE__)
void ogl_shader_check(const GLint sh, const char* file, const int line)
{
	GLint status;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &status);
	if (GL_FALSE == status)
	{
		GLint log_len;
		glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0)

		{
			std::vector<char> log(log_len, 0);
			auto written = 0;
			glGetShaderInfoLog(sh, log_len, &written, log.data());
			std::stringstream ss;
			ss << "Shader Error:" << file << ":" << line << ": " << log.data() << std::endl;
			qli_runtime_error(ss.str());
		}
		qli_runtime_error("Shader Error");
	}
}

#define OGL_PROGRAM_CHECK(ph) ogl_program_check(ph,__FILE__,__LINE__)
void ogl_program_check(const GLint ph, const char* file, const int line)
{
	GLint status;
	glGetProgramiv(ph, GL_LINK_STATUS, &status);
	if (GL_FALSE == status)
	{
		GLint log_len;
		glGetProgramiv(ph, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0)
		{
			std::vector<char> log(log_len, 0);
			auto written = 0;
			glGetProgramInfoLog(ph, log_len, &written, log.data());
			std::stringstream ss;
			ss<< "OpenGL Program Creation Error: " << file << ":" << line << ": " << log.data() << std::endl;
			qli_runtime_error(ss.str());
		}
		qli_runtime_error("OpenGL Program Creation Error");
	}
}

std::string get_last_error_as_string()
{//http://stackoverflow.com/questions/1387064/how-to-get-the-error-message-from-the-error-code-returned-by-getlasterror
 //Get the error message, if any.
	const auto error_message_id = GetLastError();
	if (error_message_id == 0)
		return std::string(); //No error message has been recorded

	LPSTR message_buffer = nullptr;
	size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, error_message_id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&message_buffer), 0, nullptr);

	std::string message(message_buffer, size);

	//Free the buffer.
	LocalFree(message_buffer);

	return message;
}

std::chrono::microseconds dvi_slm_device::vendor_stability_time() const
{
	return ms_to_chrono(50);
}
std::string dvi_slm_device_filename = "dvi_slm_device.json";

int dvi_slm_device::prompt_for_screen()
{
	std::cout << "Enter screen number:" << std::endl;
	int integer;
	while (!(std::cin >> integer))
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cout << "Invalid input.  Try again: ";
	}
	return integer;
}

dvi_slm_device::dvi_slm_device() :
	slm_device(0, 0, false), ping_(false),
	pong_(false), m_id_thread_(0),
	wcl_({ 0 }), g_h_wnd_(nullptr), g_h_instance_(nullptr), g_h_dc_(nullptr), g_h_rc_(nullptr), g_window_width_(0), g_window_height_(0), gl_done_(false), texture_(0), program_handle_(0), vao_handle_(0), frame_(0)
{
	std::ifstream configuration_file(dvi_slm_device_filename);
	if (configuration_file.is_open())
	{
		cereal::JSONInputArchive archive(configuration_file);
		archive(*this);
	}
	else
	{
		std::cout << "Warning can't find device configuration file:" << dvi_slm_device_filename << std::endl;
		preferred_monitor = prompt_for_screen();
	}
	//
	{
		lazy_wait lzw;
		window_loop_ = new std::thread(&dvi_slm_device::window_loop, this, &lzw);
	}
	{
		lazy_wait lzw;
		gl_loop_ = new std::thread(&dvi_slm_device::gl_loop, this, &lzw);
	}
}

dvi_slm_device::~dvi_slm_device()
{
	gl_done_ = true;
	set_frame_internal(0);
	gl_loop_->join();
	delete gl_loop_;
	PostThreadMessage(m_id_thread_, WM_QUIT, 0, 0);//PostQuitMessage(0);
	window_loop_->join();
	delete window_loop_;
	//
	std::ofstream os(dvi_slm_device_filename);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
		std::cout << "Writing settings file to:" << dvi_slm_device_filename << std::endl;
	}
	else
	{
		std::cout << "Warning can't write settings file to: " << dvi_slm_device_filename << std::endl;
	}
}

LRESULT CALLBACK dvi_slm_device::window_proc(const HWND h_wnd, const UINT msg, const WPARAM w_param, const LPARAM l_param)
{
	switch (msg)
	{
	case WM_CHAR:
		switch (static_cast<int>(w_param))
		{
			/*
			case VK_ESCAPE:
			PostMessage(hWnd, WM_CLOSE, 0, 0);
			break;
			*/
		default:
			break;
		}
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
		/*
		case WM_SIZE://shouldn't happen
		that->g_windowWidth = static_cast<int>(LOWORD(lParam));
		that->g_windowHeight = static_cast<int>(HIWORD(lParam));
		break;
		*/

	case WM_CREATE:
		//that->launchglLoop();
		//cout <<"Created " << endl;
		break;

	default:
		break;
	}

	return DefWindowProc(h_wnd, msg, w_param, l_param);
}

BOOL CALLBACK monitor_enum_proc(const HMONITOR h_monitor, HDC, LPRECT, const LPARAM dw_data)
{

	auto monitor_info = reinterpret_cast<std::vector<MONITORINFO>*>(dw_data);
	MONITORINFO info;
	info.cbSize = sizeof(MONITORINFO);
	GetMonitorInfo(h_monitor, &info);
	monitor_info->push_back(info);
	return TRUE;
}

void dvi_slm_device::window_loop(lazy_wait* lzw)
{
	//Application Window
	auto h_instance = GetModuleHandle(nullptr);
	wcl_.cbSize = sizeof wcl_;
	wcl_.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
	wcl_.lpfnWndProc = window_proc;
	wcl_.cbClsExtra = 0;
	wcl_.cbWndExtra = 0;
	wcl_.hInstance = g_h_instance_ = h_instance;
	wcl_.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
	wcl_.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcl_.hbrBackground = nullptr;
	wcl_.lpszMenuName = nullptr;
	wcl_.lpszClassName = L"SLM Window";
	wcl_.hIconSm = nullptr;
	RegisterClassEx(&wcl_);
	DWORD wnd_ex_style = 0;
	auto wnd_style = WS_POPUP | WS_SYSMENU;
	g_h_wnd_ = CreateWindowEx(wnd_ex_style, wcl_.lpszClassName, L"SLM", wnd_style, 0, 0, 0, 0, nullptr, nullptr, wcl_.hInstance, nullptr);
	//	assert(g_hWnd);
	//Move to monitor
	std::vector<MONITORINFO> info;
	EnumDisplayMonitors(nullptr, nullptr, monitor_enum_proc, reinterpret_cast<LPARAM>(&info));
	auto monitor = preferred_monitor < info.size() ? info[preferred_monitor] : info.front();
	//
	auto rc = monitor.rcMonitor;
	width = rc.right - rc.left;
	height = rc.bottom - rc.top;
	//MoveWindow(g_h_wnd_, rc.left, rc.top, width, height, TRUE);
	SetWindowPos(g_h_wnd_, HWND_TOPMOST, rc.left, rc.top, width, height, 0);
	GetClientRect(g_h_wnd_, &rc);
	g_window_width_ = rc.right - rc.left;
	g_window_height_ = rc.bottom - rc.top;
	//
	m_id_thread_ = GetCurrentThreadId();
	lzw->ready();
	window_event_loop();
	g_h_dc_ = nullptr;
	UnregisterClass(wcl_.lpszClassName, wcl_.hInstance);
}

int dvi_slm_device::window_event_loop()
{
	MSG msg;
	// loop until WM_QUIT(0) received
	while (GetMessage(&msg, nullptr, 0, 0) > 0)
	{
		TranslateMessage(&msg); DispatchMessage(&msg);
	}
	return static_cast<int>(msg.wParam);
}

void dvi_slm_device::gl_loop(lazy_wait* lzw)
{
	try
	{
		frame_ = 0;
		gl_done_ = false;
		//

		g_h_dc_ = GetDC(g_h_wnd_);

		// Create and set a pixel format for the window.
		PIXELFORMATDESCRIPTOR pfd;
		memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));
		pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 16;
		pfd.cDepthBits = 8;
		pfd.iLayerType = PFD_MAIN_PLANE;

		OSVERSIONINFO osvi = { 0 };
		osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
		//so it returns the sandbox version instead of the actual version, but like nobdoy cares
		// ReSharper disable once CppDeprecatedEntity
		GetVersionEx(&osvi);

		auto pf = ChoosePixelFormat(g_h_dc_, &pfd);
		SetPixelFormat(g_h_dc_, pf, &pfd);

		// When running under Windows Vista or later support desktop composition.
		// This doesn't really apply when running in full screen mode.
		if (osvi.dwMajorVersion > 6 || osvi.dwMajorVersion == 6) //-V547
		{
			pfd.dwFlags |= PFD_SUPPORT_COMPOSITION;
		}
		// Creates an OpenGL 3.1 forward compatible rendering context.
		// A forward compatible rendering context will not support any OpenGL 3.0
		// functionality that has been marked as deprecated.
		//Bootstrap glew
		auto temp_context = wglCreateContext(g_h_dc_);
		wglMakeCurrent(g_h_dc_, temp_context);
		auto foo = glewInit();
		if (foo != GLEW_OK)
		{
			const auto msg = "Oh NO! Glew Failed to Wrangle!\n";
			qli_runtime_error(msg);
		}
		int attrib_list[] =
		{
			WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
			WGL_CONTEXT_MINOR_VERSION_ARB, 1,
			WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
			0, 0
		};
		//
		wglGetExtensionsStringARB(g_h_dc_);

		//	assert(wglCreateContextAttribsARB);
		//
		g_h_rc_ = wglCreateContextAttribsARB(g_h_dc_, nullptr, attrib_list);
		wglMakeCurrent(nullptr, nullptr);
		wglDeleteContext(temp_context);
		wglMakeCurrent(g_h_dc_, g_h_rc_);
		SetWindowText(g_h_wnd_, L"SLM");
		wglSwapIntervalEXT(1);//Set the VSync?
							  //Reserve the textures and make the shaders
		const auto* vsrc1 =
			"attribute vec2 coord2d;\n"
			"attribute vec4 texCoord;\n"
			"varying vec4 texc;\n"
			"void main()\n"
			"{\n"
			"vec4 vPosition =vec4(coord2d, 0.0, 1.0);\n"
			"gl_Position = vPosition; \n"
			"texc = texCoord;\n"
			"}\n";
		GLint vs_h = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs_h, 1, &vsrc1, nullptr);
		glCompileShader(vs_h);
		OGL_SHADER_CHECK(vs_h);
		OGL_ERROR_CHECK();
		const auto* fsrc1 =
			"uniform sampler2D texture;\n"
			"varying vec4 texc;\n"
			"void main(void)\n"
			"{\n"
			"    gl_FragColor = texture2D(texture, texc.st);\n"
			"}\n";
		GLint fs_h = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs_h, 1, &fsrc1, nullptr);
		glCompileShader(fs_h);
		OGL_SHADER_CHECK(fs_h);
		OGL_ERROR_CHECK();
		//Link
		program_handle_ = glCreateProgram();
		glBindAttribLocation(program_handle_, 0, "coord2d");
		glBindAttribLocation(program_handle_, 1, "texCoord");
		glBindAttribLocation(program_handle_, 2, "texc");
		OGL_ERROR_CHECK();
		glAttachShader(program_handle_, vs_h);
		glAttachShader(program_handle_, fs_h);
		OGL_ERROR_CHECK();
		glLinkProgram(program_handle_);
		OGL_PROGRAM_CHECK(program_handle_);
		glUseProgram(program_handle_);
		OGL_ERROR_CHECK();
		//Pack static data into a VAO
		float vec_data[] = {
			-1, 1,
			-1, -1,
			1, 1,
			-1, -1,
			1, -1,
			1, 1
		};
		float tex_data[] = {
			0, 1,
			0, 0,
			1, 1,
			0, 0,
			1, 0,
			1, 1
		};
		vbo_handles_.resize(2, 0);
		glGenBuffers(2, vbo_handles_.data());
		auto vec_buf = vbo_handles_[0];
		auto tex_buf = vbo_handles_[1];
		glBindBuffer(GL_ARRAY_BUFFER, vec_buf);
		glBufferData(GL_ARRAY_BUFFER, 8 * 2 * sizeof(GLfloat), vec_data, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, tex_buf);
		glBufferData(GL_ARRAY_BUFFER, 8 * 2 * sizeof(GLfloat), tex_data, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		OGL_ERROR_CHECK();
		glGenVertexArrays(1, &vao_handle_);
		glBindVertexArray(vao_handle_);
		glEnableVertexAttribArray(0);  // Vertex position
		glEnableVertexAttribArray(1);  // Vertex color
		glBindBuffer(GL_ARRAY_BUFFER, vec_buf);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLubyte*>(nullptr));
		glBindBuffer(GL_ARRAY_BUFFER, tex_buf);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLubyte*>(nullptr)); //2 attribute per vertex
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
		OGL_ERROR_CHECK();
		//Init Textures
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &texture_);
		glBindTexture(GL_TEXTURE_2D, texture_);
		std::vector<unsigned char> hack(g_window_width_ * g_window_height_, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, g_window_width_, g_window_height_, 0, GL_RED, GL_UNSIGNED_BYTE, hack.data());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		GLint swizzle_mask[] = { GL_RED, GL_RED, GL_RED, GL_ZERO };
		glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle_mask);
		glBindTexture(GL_TEXTURE_2D, 0);
		//
		ShowWindow(g_h_wnd_, SW_MAXIMIZE);
		lzw->ready();
		while (!gl_done_)
		{
			{
				std::unique_lock<std::mutex> lk(m_);
				cv_.wait(lk, [&] {return ping_; });
			}
			ping_ = false;
			//Display Frame
			{
#if 0
				static auto tic = -1;
				tic = (1 + tic) % 2;//if you set it to tic thats UB
				glClearColor(tic, tic, tic, 0.0f);
#endif
				//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
				//
				glClear(GL_COLOR_BUFFER_BIT);
				glBindTexture(GL_TEXTURE_2D, texture_);
				auto data_ptr = frame_data_.at(frame_).data.data();
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_window_width_, g_window_height_, GL_RED, GL_UNSIGNED_BYTE, data_ptr);
				glBindVertexArray(vao_handle_);
				glDrawArrays(GL_TRIANGLES, 0, 8);
				glBindVertexArray(0);
				glBindTexture(GL_TEXTURE_2D, 0);
				glFlush();
				glFinish();
				OGL_ERROR_CHECK();
			}
			SwapBuffers(g_h_dc_); // on gdi
			{
				std::lock_guard<std::mutex> lk(m_);
				pong_ = true;
			}
			cv_.notify_one();
		}
		wglDeleteContext(g_h_rc_);
		g_h_rc_ = nullptr;
		ReleaseDC(g_h_wnd_, g_h_dc_);
	}
	catch (...)
	{
	}
}

void dvi_slm_device::set_frame_internal(const int frame_number)
{
	//Sets the frame and locks until the frame has actually changed, this is how we established synchronization 
	//Also the correct option needs to be set in the NVidia control panel specifical, vysnc should be enabled , if not forced
	frame_ = frame_number;
	//	assert(frame < framedata.size());
	{
		std::lock_guard<std::mutex> lk(m_);
		ping_ = true;
	}
	cv_.notify_one();
	{
		std::unique_lock<std::mutex> lk(m_);
		cv_.wait(lk, [&] {return pong_; });
	}
	pong_ = false;
}

void dvi_slm_device::load_frame_internal(int)
{
	//still stored in framedata
}

#endif