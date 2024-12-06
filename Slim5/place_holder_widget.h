#pragma once
#ifndef PLACE_HOLDER_WIDGET_H
#define PLACE_HOLDER_WIDGET_H
#include <QWidget>
class place_holder_widget final :public QWidget
{
	Q_OBJECT

		QSize hint_size_;
public:
	explicit place_holder_widget(QWidget* parent = nullptr) : QWidget(parent)
	{

	}
	void set_size_hint(const QSize& size_hint)
	{
		if (hint_size_ != size_hint)
		{
			hint_size_ = size_hint;
			updateGeometry();
		}
	}

	[[nodiscard]] QSize sizeHint() const override
	{
		return hint_size_;
	}
};
#endif