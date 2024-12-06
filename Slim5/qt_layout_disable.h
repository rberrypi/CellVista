#pragma once
#ifndef LAYOUT_DISABLE_H
#define LAYOUT_DISABLE_H
#include <QLayout>
#include <QWidget>

inline void enable_layout(QLayout* layout,const  bool enable)
{
	const auto item_count = layout->count();
		for (auto item_idx = 0; item_idx < item_count; ++item_idx)
		{
			auto* widget_handle =layout->itemAt(item_idx)->widget();
			if (widget_handle)
			{
				widget_handle->setEnabled(enable);
			}
		}
}

#endif