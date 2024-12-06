#include "stdafx.h"
#include "aspectratiopixmaplabel.h"
//#include <QDebug>

AspectRatioPixmapLabel::AspectRatioPixmapLabel(QWidget *parent) :
	QLabel(parent)
{
	this->setMinimumSize(1, 1);
	setScaledContents(false);
}

void AspectRatioPixmapLabel::setPixmap(const QPixmap & p)
{
	pix = p;
	QLabel::setPixmap(scaledPixmap());
	//QLabel::setPixmap(pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

int AspectRatioPixmapLabel::heightForWidth(int width) const
{
	return pix.isNull() ? this->height() : ((qreal)pix.height()*width) / pix.width();
}

QSize AspectRatioPixmapLabel::sizeHint() const
{
	const auto w = this->width();
	return QSize(w, heightForWidth(w));
}

QPixmap AspectRatioPixmapLabel::scaledPixmap() const
{
	//return pix.scaled(this->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	auto scaled = pix.scaled(this->size() * devicePixelRatioF(), Qt::KeepAspectRatio, Qt::SmoothTransformation); scaled.setDevicePixelRatio(devicePixelRatioF());
	return scaled;
}

void AspectRatioPixmapLabel::resizeEvent(QResizeEvent * e)
{
	if (!pix.isNull())
		QLabel::setPixmap(scaledPixmap());
}