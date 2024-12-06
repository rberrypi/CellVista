#include "stdafx.h"
#include <QSpinBox>
#include <QPainter>
#include <sstream> 
#include "scope_delay_delegate.h"
#include <QHBoxLayout>

#include "capture_item.h"


QWidget* scope_delay_delegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<scope_delays>())
	{
		const auto as_delays = as_variant.value<scope_delays>();
		auto* roi_delay = new QSpinBox(parent);

		const auto max_int = std::numeric_limits<int>::max();
		roi_delay->setButtonSymbols(QAbstractSpinBox::NoButtons);
		roi_delay->setAlignment(Qt::AlignCenter);
		roi_delay->setSuffix(" ms");
		roi_delay->setMaximum(max_int);
		roi_delay->setToolTip("Delay after acquiring the ROI [ms]");

		const auto roi_move_delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(as_delays.roi_move_delay);
		roi_delay->setValue(roi_move_delay_ms.count());

		return roi_delay;
	}
	return QStyledItemDelegate::createEditor(parent, option, index);
}

void scope_delay_delegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<scope_delays>())
	{
		const auto roi_delay = qobject_cast<QSpinBox*>(editor);
		const auto as_delays = as_variant.value<scope_delays>();
		const auto roi_move_delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(as_delays.roi_move_delay);
		roi_delay->setValue(roi_move_delay_ms.count());

	}
	QStyledItemDelegate::setEditorData(editor, index);
}

void scope_delay_delegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<scope_delays>())
	{
		const auto cmb = qobject_cast<QSpinBox*>(editor);
		scope_delays delays;
		delays.roi_move_delay = ms_to_chrono(cmb->value());
		const auto new_value_as_variant = QVariant::fromValue(delays);
		model->setData(index, new_value_as_variant);
	}
	else
	{
		QStyledItemDelegate::setModelData(editor, model, index);
	}
}

template<typename T, typename V> T& display_time_two(T& os, V ns)
{
	auto fill = os.fill();
	os.fill('0');
	auto d = std::chrono::duration_cast<std::chrono::duration<int, std::ratio<86400>>>(ns);
	ns -= d;
	auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
	ns -= h;
	auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
	ns -= m;
	auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
	ns -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns);
	ns -= ms;
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(ns);
	//ns -= us;
	if (d.count() > 0)
	{
		os << std::setw(2) << d.count() << "d";
	}
	if (h.count())
	{
		if (d.count() > 0)
		{
			os << ":";
		}
		os << std::setw(2) << h.count() << "h";
	}
	if (m.count())
	{
		if (h.count() > 0)
		{
			os << ":";
		}
		os << std::setw(2) << m.count() << "m";
	}
	if (s.count() > 0)
	{
		if (m.count() > 0)
		{
			os << ":";
		}
		os << std::setw(2) << s.count() << "s";
	}
	if (ms.count() > 0)
	{
		if (s.count() > 0)
		{
			os << ":";
		}
		os << std::setw(3) << ms.count() << "ms";
	}
	if (us.count() > 0)
	{
		if (ms.count() > 0)
		{
			os << ":";
		}
		os << std::setw(3) << us.count() << "us";
	}
	os.fill(fill);
	return os;
}

void scope_delay_delegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<scope_delays>())
	{
		const auto as_delays = as_variant.value<scope_delays>();
		const auto  font = index.data(Qt::FontRole).value<QFont>();
		std::stringstream roi_delay_ss;
		display_time_two(roi_delay_ss, as_delays.roi_move_delay);
		auto label_roi = QString::fromStdString(roi_delay_ss.str());

		if (label_roi.isEmpty())
		{
			label_roi.append("none");
		}
		painter->save();
		painter->setFont(font);
		painter->drawText(option.rect, Qt::AlignCenter, label_roi);
		painter->restore();
	}
	else
	{
		QStyledItemDelegate::paint(painter, option, index);
	}
}
