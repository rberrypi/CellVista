#include "stdafx.h"
#include "rectangle_model.h"

void rectangle_model::fill_column(const QVariant& value, const int column_index)
{
	const auto rows = rowCount();
	for (auto i = 0; i < rows; ++i)
	{
		auto idx = createIndex(i, column_index);
		setData(idx, value);
	}
}

void rectangle_model::resize_to(const int new_size)
{
	const auto change = new_size - rowCount();
	if (change > 0)
	{
		insertRows(rowCount(), change);
	}
	if (change < 0)
	{
		removeRows(rowCount() - 1, abs(change));
	}
}

