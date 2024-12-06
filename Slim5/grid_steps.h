#pragma once
#ifndef GRID_STEPS_H
#define GRID_STEPS_H

struct grid_steps final
{
	double x_step, y_step;
	int x_steps, y_steps;

	[[nodiscard]] int count() const noexcept
	{
		return x_steps * y_steps;
	}
	grid_steps() noexcept: grid_steps(0, 0, 0, 0) {}

	[[nodiscard]] grid_steps grow_by_one() const noexcept
	{
		auto steps = *this;
		steps.x_steps += 1;
		steps.y_steps += 1;
		return steps;
	}
	grid_steps(const double x_step, const double y_step, const int x_steps, const int y_steps) noexcept:
		x_step(x_step), y_step(y_step), x_steps(x_steps), y_steps(y_steps)
	{}
};

#endif
