#include "stdafx.h"
#include "cgal_triangulator.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>

#include "qli_runtime_error.h"

struct k : CGAL::Exact_predicates_inexact_constructions_kernel {};
typedef CGAL::Delaunay_triangulation_2<k> triangulation;
typedef triangulation::Vertex_handle vertex_handle;
typedef k::FT coordinate_type;
typedef k::Point_2 point;
typedef CGAL::Data_access< std::map<point, coordinate_type, k::Less_xy_2 > > value_access;

struct value_duplication_pair final
{
	coordinate_type value;
	int aliases;
};

struct cgal_triangulation_holder final
{
	triangulation t;
	std::map<point, value_duplication_pair, k::Less_xy_2> function_values;
	void check_triangulation() const
	{
#if _DEBUG
		auto in_values = 0;
		// ReSharper disable once CppDeclaratorNeverUsed
		for (auto value : function_values)
		{
			in_values += 1;
		}
		const auto in_points = std::distance(t.points_begin(), t.points_end());
		if (in_values != in_points)
		{
			qli_runtime_error("Triangulation has gone out of sync!");
		}
#endif
	}

	[[nodiscard]] int get_total_points() const
	{
		auto unique_points = 0;
		for (const auto& value : function_values)
		{
			unique_points += (value.second.aliases + 1);
		}
		return unique_points;
	}
	void debug_triangulation(const std::string& label, const bool force_debug = false) const
	{
		if ( force_debug)
		{
			const auto caption = "==== " + label + " ===";
			std::cout << caption << std::endl;
			const auto points = std::distance(t.points_begin(), t.points_end());
			std::cout << "Points: " << points << std::endl;
			for (auto it = t.points_begin(); it != t.points_end(); ++it)
			{
				std::cout << *it << std::endl;
			}
			std::cout << "Values: " << function_values.size() << std::endl;
			std::cout << "Total: " << get_total_points() << std::endl;
			for (auto pairs : function_values)
			{
				std::cout << pairs.first << "->[" << pairs.second.value << "," << pairs.second.aliases << "]" << std::endl;
			}
			std::cout << "Is Valid? " << t.is_valid(true) << std::endl;
			for (size_t i = 0; i < caption.size(); ++i)
			{
				std::cout << "=";
			}
			std::cout << std::endl;
		}
	}
};

int cgal_triangulator::get_total_points() const
{
	return holder->get_total_points();
}

float cgal_triangulator::interpolate_one(const scope_location_xy& loc) const
{
	const point p(loc.x, loc.y);
	static std::vector<std::pair<point, coordinate_type>> coords(5);
	coords.resize(0);
	const auto result = natural_neighbor_coordinates_2(holder->t, p, std::back_inserter(coords));
	if (result.third)
	{
		const auto norm = result.second;
		auto first = coords.begin();
		const auto beyond = coords.end();
		coordinate_type interpolation = 0;
		{
			for (; first != beyond; ++first) {
				const auto val = this->holder->function_values.at(first->first).value;
				interpolation += (first->second / norm) * val;
			}
		}
		return interpolation;
	}
	return std::numeric_limits<float>::quiet_NaN();
}

void cgal_triangulator::interpolate(std::vector<scope_location_xyz>& query_points) const
{
	holder->debug_triangulation("Interpolate", false);
	for (auto& query : query_points)
	{
		query.z = interpolate_one(query);
	}
	holder->check_triangulation();
}

void cgal_triangulator::interpolate(std::vector<interpolation_pair>& query_points) const
{

	for (auto& query : query_points)
	{
		query.second.z = interpolate_one(query.second);
	}
	holder->check_triangulation();
}

void cgal_triangulator::remove_point(const scope_location_xyz& loc)
{
	const point loc_as_point = { loc.x, loc.y };
	auto handle = holder->function_values.find(loc_as_point);
#if _DEBUG
	{
		const auto cant_find_point = handle == holder->function_values.end();
		if (cant_find_point)
		{
			qli_runtime_error("Can't find point");
		}
	}
#endif
	const auto has_alias = handle->second.aliases > 0;
	if (has_alias)
	{
		handle->second.aliases -= 1;
	}
	else
	{
		holder->function_values.erase(loc_as_point);
		const auto handle_t = holder->t.nearest_vertex(loc_as_point);
		holder->t.remove(handle_t);
	}
	holder->debug_triangulation("After Remove Point");
	holder->check_triangulation();
}

bool cgal_triangulator::insert_point(const scope_location_xyz& loc)
{
	const point loc_as_point = { loc.x, loc.y };
	auto handle = holder->function_values.find(loc_as_point);
	const auto is_duplicate = handle != holder->function_values.end();
	if (is_duplicate)
	{
		handle->second.aliases += 1;
	}
	else
	{
		holder->t.insert(loc_as_point);//if alias won't do nothing
		holder->function_values.insert({ loc_as_point ,{ loc.z,0 } });
	}
	holder->debug_triangulation("After Point Insert");
	holder->check_triangulation();
	return is_duplicate;
}

bool cgal_triangulator::move_point(const scope_location_xyz& src, const scope_location_xyz& dst)
{
#if _DEBUG
	{
		const auto missing = holder->function_values.find({ src.x,src.y }) == holder->function_values.end();
		if (missing)
		{
			qli_runtime_error("Somehow this point slipped out");
		}
	}
	const auto points_at_start = holder->get_total_points();
#endif
	const auto is_zee_only = static_cast<const scope_location_xy&>(src) == static_cast<const scope_location_xy&>(dst);
	if (is_zee_only)
	{
		const auto handle = holder->function_values.find({ src.x,src.y });
		handle->second.value = dst.z;
		const auto move = "Zee only move [" + std::to_string(src.x) + "," + std::to_string(src.y) + "," + std::to_string(src.z) + "] -> [" + std::to_string(dst.x) + "," + std::to_string(dst.y) + "," + std::to_string(dst.z) + "]";
		holder->debug_triangulation(move);
		return handle->second.aliases > 0;
	}
	const auto is_alias = [&](const scope_location_xyz& loc)
	{
		const auto item = this->holder->function_values.find({ loc.x,loc.y });
		if (item == this->holder->function_values.end())
		{
			return false;
		}
		return item->second.aliases > 0;
	};
	const auto will_alias = [&](const scope_location_xyz& loc)
	{
		const auto item = this->holder->function_values.find({ loc.x,loc.y });
		return  item != this->holder->function_values.end();
	};
	const auto modify_count = [&](const scope_location_xyz& loc, const int inc)
	{
		holder->function_values.find({ loc.x,loc.y })->second.aliases += inc;
#if _DEBUG
		if (holder->function_values.find({ loc.x,loc.y })->second.aliases < 0)
		{
			qli_runtime_error("Lol wut, don't do this");
		}
#endif
	};
	const auto src_alias = is_alias(src);
	const auto dst_alias = will_alias(dst);
	if (!src_alias && !dst_alias)
	{
		holder->function_values.erase({ src.x,src.y });
		const auto handle = holder->t.nearest_vertex({ src.x,src.y });
		holder->t.move(handle, { dst.x,dst.y });
		holder->function_values.insert({ { dst.x,dst.y }, { dst.z, 0 } });
	}
	else if (src_alias && !dst_alias)
	{
		modify_count(src, -1);
		holder->function_values.insert({ { dst.x,dst.y },{ dst.z, 0 } });
		holder->t.insert({ dst.x,dst.y });
	}
	else if (src_alias && dst_alias)
	{
		modify_count(src, -1);
		modify_count(dst, 1);
	}
	else if (!src_alias && dst_alias)
	{
		holder->function_values.erase({ src.x, src.y });
		const auto handle = holder->t.nearest_vertex({ src.x, src.y });
		holder->t.remove(handle);
		modify_count(dst, 1);
	}
	const auto move = "After Point Move [" + std::to_string(src.x) + "," + std::to_string(src.y) + "] -> [" + std::to_string(dst.x) + ", " + std::to_string(dst.y) + "]";
	holder->debug_triangulation(move);
	holder->check_triangulation();
#if _DEBUG
	const auto points_at_end = holder->get_total_points();
	if (points_at_start != points_at_end)
	{
		qli_runtime_error("Somehow you lost a point, which is bad, yo");
	}
#endif
	return dst_alias;
}

float cgal_triangulator::get_z(const scope_location_xy& loc) const
{
	return holder->function_values.at({ loc.x,loc.y }).value;
}

cgal_triangulator::cgal_triangulator()
{
	holder = std::make_unique<cgal_triangulation_holder>();
}

cgal_triangulator::~cgal_triangulator() = default;
