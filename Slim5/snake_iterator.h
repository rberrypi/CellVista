#pragma once
#ifndef SNAKE_ITERATOR_H
#define SNAKE_ITERATOR_H

struct snake_iterator final
{
	struct column_row_pair final
	{
		int column;
		int row;
	};

	static column_row_pair iterate(const int i, const int c) noexcept//really bad programming but its like 6:30 AM
	{
		const auto row = static_cast<int>(i / c);
		const auto rem = i % c;
		const auto col = row % 2 == 0 ? rem : c - 1 - rem;
		return{ col,row };
	}

	static int count(const int columns, const int rows) noexcept
	{
		return columns * rows;
	}
};

#endif