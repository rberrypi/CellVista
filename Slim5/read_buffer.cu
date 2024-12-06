#include "write_tif.h"
#include <tiffio.h>
#include <iostream>
#include "qli_runtime_error.h"

template <typename T>
void read_buffer(const std::string& fname, tiff_image<T>& ret)//deallocate with 
{
	//todo why not fill an std::vector?
	const auto tif = TIFFOpen(fname.c_str(), "r");//tiff open catch error?
	if (tif == nullptr)
	{
		const auto error_str = "Can't get file: " + std::string(fname);//insert token to continue
		qli_runtime_error(error_str);
	}
	//quick and dirty dircount
	uint32 img_h, img_w;
	uint16 bits_per_sample, samples_per_pixel;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img_h);
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &img_w);
	const auto tagsuccess = TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
	if ((tagsuccess == 1) && (bits_per_sample != sizeof(T) * 8))
	{
		const auto error_str = "Wrong bit depth: " + std::string(fname);//insert token to continue
		qli_runtime_error(error_str);
	}
	uint16 config_should_be_planner;
	TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &config_should_be_planner);
	TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
	const auto expected_row_size = samples_per_pixel * img_w * sizeof(T);
	const auto row_size = TIFFScanlineSize(tif);
	if (expected_row_size != row_size)
	{
		qli_not_implemented();
	}
	const auto pixels = img_w * img_h * samples_per_pixel;
	ret.img.resize(pixels);
	ret.samples_per_pixel = samples_per_pixel;
	ret.width = img_w;
	ret.height = img_h;
	const auto mydata = reinterpret_cast<unsigned char*>(ret.img.data());
	for (auto row = 0; row < img_h; row++)
	{
		// ReSharper disable once CppLocalVariableMayBeConst
		auto to_me = static_cast<void*>(&mydata[row_size * row]);
		TIFFReadScanline(tif, to_me, row);
	}
	TIFFClose(tif);
}

template <typename T>
static tiff_image<T> read_buffer(const std::string& file_name)//deallocate with 
{
	tiff_image<T> ret;
	read_buffer(file_name, ret);
	return ret;
}