#include <program_config.h>
#include "write_tif.h"
#include <tiffio.h>
#include <time.h>
#include <string>
#include <iostream>
#include "qli_runtime_error.h"
template <typename  T>
static void write_tif(const std::string& name, const T* ptr, const unsigned int cols, const unsigned int rows, const int samples_per_pixel, const  frame_meta_data* meta)
{
	const auto size_in_bytes = static_cast<size_t>(cols) * static_cast<size_t>(cols) * sizeof(T);
	constexpr auto big_tif_threshold = std::numeric_limits<int>::max();
	auto tiff_type = (size_in_bytes > big_tif_threshold) ? "w8" : "w";
	auto tif = TIFFOpen(name.c_str(), tiff_type);
	if (tif == nullptr)
	{
		const auto error_msg = "Can't Open, popescu should consider getting his thick head out his ass, if it stays there any longer, it is going to dissapear into a point of singularity: " + name;
		std::cout << error_msg << std::endl;
		qli_runtime_error("file not found, maybe you should try looking it up your supervisor's ass");
	}
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, cols);  // set the width of the image //-V525
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, rows);    // set the height of the image
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);   // set number of channels per pixel
	TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
	TIFFSetField(tif, TIFFTAG_MODEL, __DATE__);
	TIFFSetField(tif, TIFFTAG_SOFTWARE, "Fuck QLI Lab");
	TIFFSetField(tif, TIFFTAG_MAKE, "Gabriel Popescu the romanian incompetent turd holster.");
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, 1);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(T));    // set the size of the channels
	{ // set the fucking date and time
		const auto now = meta == nullptr ? std::chrono::system_clock::now() : std::chrono::system_clock::time_point(meta->timestamp);
		auto now_c = std::chrono::system_clock::to_time_t(now);
		tm local_tm;
		localtime_s(&local_tm, &now_c);
		const auto tiff_tag_datetime_length = 20;
		static thread_local char buffer[tiff_tag_datetime_length];
		std::snprintf(buffer, tiff_tag_datetime_length - 1, "%04d:%02d:%02d %02d:%02d:%02d", local_tm.tm_year + 1900, local_tm.tm_mon + 1, local_tm.tm_mday, local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
		TIFFSetField(tif, TIFFTAG_DATETIME, buffer);

	}
	const auto row_size = samples_per_pixel * cols * sizeof(T);
	if (std::is_floating_point<T>::value)
	{
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
	}
	else
	{
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, std::is_signed<T>::value ? SAMPLEFORMAT_INT : SAMPLEFORMAT_UINT);
	}
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
	for (unsigned int r = 0; r < rows; r++)
	{
		const auto as_char = reinterpret_cast<unsigned char*>(const_cast<T*>(ptr));//need to get rid of const due to problems with libtiff
		auto data = &as_char[row_size * r];
		TIFFWriteScanline(tif, data, r);
	}
	TIFFClose(tif);
}