#pragma once
#ifndef CGAL_TRIANGULATOR_H
#define CGAL_TRIANGULATOR_H
#include "snake_iterator.h"
#include "instrument_configuration.h"
#include <memory>
struct cgal_triangulation_holder;
class cgal_triangulator final
{
public:
	[[nodiscard]] int get_total_points() const;
	std::unique_ptr<cgal_triangulation_holder> holder;
	typedef std::pair<snake_iterator::column_row_pair, scope_location_xyz> interpolation_pair;
	void interpolate(std::vector<scope_location_xyz>& query_points) const;
	void interpolate(std::vector<interpolation_pair>& query_points) const;
	void remove_point(const scope_location_xyz& loc);
	bool insert_point(const scope_location_xyz& loc);
	bool move_point(const scope_location_xyz& src, const scope_location_xyz& dst);
	[[nodiscard]] float get_z(const scope_location_xy& loc) const;
	[[nodiscard]] float interpolate_one(const scope_location_xy& loc) const;
	cgal_triangulator();
	~cgal_triangulator();
};
#endif