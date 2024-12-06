#include "stdafx.h"
#include "slm_device.h"
#include <QImage>
#include <QDir>
#include <iostream>
#include "write_tif.h"

void slm_device::fill_custom_pattern(unsigned char* buff, const std::filesystem::path& full_path, const frame_size& slm_size)
{
	const auto full_filename = full_path.string();
	const auto filename = QString::fromStdString(full_filename);
	const QImage img(filename);
	const auto has_image = !img.isNull();
	const auto is_correct_format = img.isGrayscale();
	const auto is_correct_size = (img.size().height() == slm_size.height) && (img.size().width() == slm_size.width);
	const auto load_image = has_image && is_correct_format && is_correct_size;
	if (load_image)
	{
		const auto ptr_img = static_cast<const unsigned char*>(img.constBits());
		std::copy(ptr_img, ptr_img + slm_size.n(), buff);
		std::cout << "Loaded: " << full_filename << std::endl;
	}
	std::cout << "Failed to load: " << full_filename << std::endl;
	std::fill(buff, buff + slm_size.n(), 0u);
}
