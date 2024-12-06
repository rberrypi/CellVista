#include "stdafx.h"
#include "slm_device.h"
#include <QSvgRenderer> 
#include <QPainter>
#include "qli_runtime_error.h"

//Todo replace with modulo when the brain starts working
void slm_device::fill_symbol(unsigned char* buff, const unsigned char outer, const unsigned char inner, const frame_size& slm_size)
{
	if (alignment_patterns.isNull())
	{
		alignment_patterns = [&] {
			QSvgRenderer renderer(QString(":/images/USAF-1951.svg"));
			if (!renderer.isValid())
			{
				qli_invalid_arguments();
			}
			QImage image(slm_size.width, slm_size.height, QImage::Format_ARGB32);
			image.fill(255);
			QPainter painter(&image);
			renderer.render(&painter);
			return image.convertToFormat(QImage::Format_Grayscale8);
		}();
	}
	const auto b = alignment_patterns.bits();
	//std::copy(b, b + slm_size.n(), buff);
	const auto functor = [&](const uchar mask) {return mask == 0 ? inner : outer; };
	std::transform(b, b + slm_size.n(), buff, functor);
}