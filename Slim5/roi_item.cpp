#include "stdafx.h"
#include "roi_item.h"
#include <QPainter>
#include "device_factory.h"
#include "scope.h"
#include <QGraphicsItem>
#include "roi_model.h"
#include <boost/align/aligned_allocator.hpp>


float roi_item::zee_max_global = -std::numeric_limits<float>::infinity();
float roi_item::zee_min_global = std::numeric_limits<float>::infinity();
bool roi_item::do_interpolation_global_ = false;
bool roi_item::do_interpolation_local_ = false;

roi_item::roi_item(const int id, roi_item_serializable& data, QGraphicsItem* parent) : roi_item_shared(data), roi_item_dimensions(data), QAbstractGraphicsShapeItem(parent), id_(id), steps_(grid_steps(0, 0, 0, 0)), grid_center_(static_cast<scope_location_xy>(D->scope->get_state())), /*grid_selected_(false),*/xyz_model(this)
{
	set_x(data.x);
	set_y(data.y);
	update_grid();
}

float roi_item::get_x() const
{
	return boundingRect().center().x();
}

std::vector<float> roi_item::get_z_all_focus_points() const
{
	std::vector<float> z_values;
	const auto rows = xyz_model.rowCount();
	for (auto row = 0; row < rows; ++row)
	{
		z_values.push_back(xyz_model.get_focus_point_location(row).z);
	}
	return z_values;
}

void roi_item::set_all_focus_points(const float zee_level) //const
{
	const auto rows = xyz_model.rowCount();
	for (auto row = 0; row < rows; ++row)
	{
		xyz_model.setData(xyz_model.index(row, XYZ_MODEL_ITEM_Z_IDX), zee_level);

	}
}

void roi_item::set_all_focus_points(const std::vector<float>& z_values)
{
	const auto rows = xyz_model.rowCount();
#if DEBUG
	if (rows != z_values.size())
	{
		qli_runtime_error("PROBLEM!!");
	}
#endif
	for (auto row = 0; row < rows; ++row)
	{
		xyz_model.setData(xyz_model.index(row, XYZ_MODEL_ITEM_Z_IDX), z_values[row]);

	}
}

void roi_item::increment_all_focus_points(const float increment)// const
{
	const auto rows = xyz_model.rowCount();
	for (auto row = 0; row < rows; ++row)
	{
		const auto current_value = xyz_model.data(xyz_model.index(row, XYZ_MODEL_ITEM_Z_IDX), Qt::DisplayRole).value<float>();
		const auto final_value = current_value + increment;
		xyz_model.setData(xyz_model.index(row, XYZ_MODEL_ITEM_Z_IDX), final_value);
	}
}

xy_pairs roi_item::get_xy_focus_points() const
{
	xy_pairs xy_vector;
	for (auto i = xyz_focus_points_model::min_rows; i < xyz_model.rowCount(); ++i)
	{
		auto x = xyz_model.data(xyz_model.index(i, XYZ_MODEL_ITEM_X_IDX), Qt::DisplayRole).value<float>();
		auto y = xyz_model.data(xyz_model.index(i, XYZ_MODEL_ITEM_Y_IDX), Qt::DisplayRole).value<float>();
		xy_vector.emplace_back(x, y);
	}
	return xy_vector;
}
void roi_item::set_xy_focus_points(const xy_pairs& xy_focus_points, const float displacement_x, const float displacement_y)
{
	const auto rows = xyz_focus_points_model::min_rows;
	for (auto i = 4; i < xyz_model.rowCount(); ++i)
	{
		const auto& pair = xy_focus_points.at(i-rows);
		xyz_model.setData(xyz_model.index(i, XYZ_MODEL_ITEM_X_IDX), pair.first - displacement_x, Qt::EditRole);
		xyz_model.setData(xyz_model.index(i, XYZ_MODEL_ITEM_Y_IDX), pair.second - displacement_y, Qt::EditRole);
	}
}

/*
1) get bounding rect
2) set and old rect
3) set new rect center only (does not change width and height, only location)
4) need to set focus points centers somehow
5) update() for current new rec
6) update(old) rec
 */
void roi_item::set_x(const float x)
{
	const auto old_rect = boundingRect();
	const scope_location_xy center(x, old_rect.center().y());
	set_center(center);
	update_grid();

	update();
	update(old_rect);
}

float roi_item::get_y() const
{
	return boundingRect().center().y();
}

void roi_item::set_y(const float y)
{
	const auto old_rect = boundingRect();
	const scope_location_xy center(old_rect.center().x(), y);
	set_center(center);
	update_grid();

	update();
	update(old_rect);
}

void roi_item::update_grid()
{
	const auto old_rect = boundingRect();

	const auto center = get_grid_center();
	const auto steps = get_roi_steps();
	xyz_model.update_four_points(steps, center);
	set_grid_steps(steps);

	update();
	update(old_rect);
}

void roi_item::set_center(const scope_location_xy& center)
{
	set_grid_center(center);
	update_grid();
}

grid_steps roi_item::get_roi_steps() const
{
	const auto x_steps = get_columns();
	const auto y_steps = get_rows();
	const auto x_step = get_column_step();
	const auto y_step = get_row_step();
	return { x_step, y_step, x_steps, y_steps };
}

void roi_item::set_delays(const scope_delays& delays)
{
	roi_move_delay = delays.roi_move_delay;
}


roi_item_serializable roi_item::get_roi_item_serializable() const
{
	const auto location = scope_location_xy(get_x(), get_y());
	const auto roi_focus_points = xyz_model.get_serializable_focus_points();

	roi_item_serializable item(*this, location, *this, roi_focus_points);
	return item;
}


bool roi_item::focus_points_verified() const
{
	for (auto& focus_point : xyz_model.data_view_)
	{
		if (!focus_point->get_verified())
		{
			return false;
		}
	}
	return true;
}

void roi_item::verify_focus_points(const bool verify)
{
	const auto rows = xyz_model.rowCount();
	for (auto row = 0; row < rows; ++row)
	{
		xyz_model.setData(xyz_model.index(row, XYZ_MODEL_ITEM_VALID_IDX), verify);
	}
}

void roi_item::set_roi_item_serializable(const roi_item_serializable& serializable)
{
	static_cast<roi_item_shared&>(*this) = serializable;
	static_cast<roi_item_dimensions&>(*this) = serializable;
	xyz_model.set_serializable_focus_points(serializable.focus_points);
	set_center(scope_location_xy(serializable.x, serializable.y));
	//grid_selected(serializable.grid_selected_);
	//grid_selected(false);
}

void roi_item_serializable::merge_b_into_a(roi_item_serializable& a, const roi_item_serializable& b)
{
	//combine the xy hulls of these guys
	const auto left = std::min(a.x - a.column_step * a.columns / 2, b.x - b.column_step * b.columns / 2);
	const auto right = std::max(a.x + a.column_step * a.columns / 2, b.x + b.column_step * b.columns / 2);
	const auto bot = std::max(a.y + a.row_step * a.rows / 2, b.y + b.row_step * b.rows / 2);
	const auto top = std::min(a.y - a.row_step * a.rows / 2, b.y - b.row_step * b.rows / 2);
	a.x = (left + right) / 2;
	a.y = (top + bot) / 2;
	a.columns = ceil((right - left) / a.column_step);
	a.rows = ceil((bot - top) / a.row_step);
}

QRectF roi_item::boundingRect() const
{
	const auto left = grid_center_.x - (steps_.x_steps * steps_.x_step) / 2;
	const auto top = grid_center_.y - (steps_.y_steps * steps_.y_step) / 2;
	const auto width = steps_.x_step * steps_.x_steps;
	const auto height = steps_.y_step * steps_.y_steps;
	return { left, top, width, height };
}

void roi_item::set_grid_steps(const grid_steps& steps)
{
	steps_ = steps;
	update();
}

void roi_item::set_grid_center(const scope_location_xy& center)
{
	grid_center_ = center;
	update();
}

void roi_item::grid_selected(const bool selected)
{
	grid_selected_ = selected;
	xyz_model.set_selection_color(selected);

}

float roi_item::query_triangulator(const scope_location_xy& point) const
{
	return triangulator.interpolate_one(point);
}

void roi_item::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	painter->save();
	const QPen pen_grid(Qt::black, 5, Qt::SolidLine);
	painter->setPen(pen_grid);
	//Checker board
	if (do_interpolation_local_ || do_interpolation_global_)
	{
		static std::vector<scope_location_xyz> query_points;
		const auto n = steps_.count();
		query_points.reserve(n);
		query_points.resize(0);
		const auto grid_left_offset = grid_center_.x - (steps_.x_step * steps_.x_steps / 2) + (steps_.x_step / 2);
		const auto grid_top_offset = grid_center_.y - (steps_.y_step * steps_.y_steps / 2) + (steps_.y_step / 2);
		const auto  round_up = [](const int num_to_round, const int multiple)
		{
			return (num_to_round + multiple - 1) / multiple * multiple;
		};
		const auto aligned_to_32_bit_boundary = round_up(steps_.x_steps, 4);
		for (auto row = 0; row < steps_.y_steps; ++row)
		{
			for (auto col = 0; col < aligned_to_32_bit_boundary; ++col)
			{
				const float x = grid_left_offset + col * steps_.x_step;
				const float y = grid_top_offset + row * steps_.y_step;
				const auto blank = std::numeric_limits<float>::quiet_NaN();
				query_points.emplace_back(x, y, blank);
			}
		}
		triangulator.interpolate(query_points);

		auto zee_min_local = std::numeric_limits<float>::infinity();
		auto zee_max_local = -std::numeric_limits<float>::infinity();
		for (const auto& element : query_points)
		{
			if (isfinite(element.z))
			{
				zee_min_local = std::min(zee_min_local, element.z);
				zee_max_local = std::max(zee_max_local, element.z);

				zee_min_global = std::min(zee_min_global, element.z);
				zee_max_global = std::max(zee_max_global, element.z);
			}
		}

		//default interpolation done in each ROI locally
		auto zee_min = zee_min_local;
		auto zee_max = zee_max_local;

		//Sometimes want interpolation done between all ROI 's (globally)
		if (do_interpolation_global_)
		{
			zee_min = zee_min_global;
			zee_max = zee_max_global;
		}

		if (zee_min != zee_max)
		{

			const auto hsl_to_rgb = [&](const int hue, const float saturation, const float lightness)
			{
				unsigned char red;
				unsigned char green;
				unsigned char blue;

				if (saturation == 0)
				{
					red = green = blue = static_cast<unsigned char>(lightness * 255);
				}
				else
				{
					const float v2 = (lightness < 0.5) ? (lightness * (1 + saturation)) : ((lightness + saturation) - (lightness * saturation));
					const float v1 = 2 * lightness - v2;

					const auto hue_scale = static_cast<float>(hue) / 360;

					const auto hue_to_rgb = [&](const float val1, const float val2, float val_h)
					{
						if (val_h < 0)
						{
							val_h += 1;
						}

						if (val_h > 1) {
							val_h -= 1;
						}

						if ((6 * val_h) < 1) {
							return (val1 + (val2 - val1) * 6 * val_h);
						}
						if ((2 * val_h) < 1) {
							return val2;
						}

						if ((3 * val_h) < 2) {
							return (val1 + (val2 - val1) * ((2.0f / 3) - val_h) * 6);
						}
						return val1;
					};
					red = static_cast<unsigned char>(255 * hue_to_rgb(v1, v2, hue_scale + (1.0f / 3)));
					green = static_cast<unsigned char>(255 * hue_to_rgb(v1, v2, hue_scale));
					blue = static_cast<unsigned char>(255 * hue_to_rgb(v1, v2, hue_scale - (1.0f / 3)));
				}

				return rgb{ red, green, blue };
			};


			static std::vector<uchar, boost::alignment::aligned_allocator<uchar, 4>> rgb_query_points_scaled;
			rgb_query_points_scaled.resize(0);

			const auto scale_factor_hsl = (1.0f) / (zee_max - zee_min);
			for (const auto& location : query_points)
			{
				const auto pixel_value = (location.z - zee_min) * scale_factor_hsl;
				const auto intensity_hsl = std::min(1.0f, pixel_value);		//range: 0~1

				const auto red = grid_selected_ ? hsl_to_rgb(190, 1.0f, intensity_hsl).red : intensity_hsl * 255.0;
				const auto green = grid_selected_ ? hsl_to_rgb(190, 1.0f, intensity_hsl).green : intensity_hsl * 255.0;
				const auto blue = grid_selected_ ? hsl_to_rgb(190, 1.0f, intensity_hsl).blue : intensity_hsl * 255.0;
				const auto alpha = 150.0f;

				rgb_query_points_scaled.push_back(red);
				rgb_query_points_scaled.push_back(green);
				rgb_query_points_scaled.push_back(blue);
				rgb_query_points_scaled.push_back(alpha);
			}

			const auto bytes_per_line = sizeof(unsigned char) * 4 * aligned_to_32_bit_boundary;
			const QImage draw_me(rgb_query_points_scaled.data(), steps_.x_steps, steps_.y_steps, bytes_per_line, QImage::Format_RGBA8888);
			const auto left = grid_center_.x - (steps_.x_step * steps_.x_steps / 2);
			const auto top = grid_center_.y - (steps_.y_step * steps_.y_steps / 2);
			const QRectF target(left, top, steps_.x_step * steps_.x_steps, steps_.y_step * steps_.y_steps);
			painter->drawImage(target, draw_me);
		}
	}
	//Around the grid
	//Grid
	{
		QVector<QLineF> lines;
		const auto left = grid_center_.x - (steps_.x_step * steps_.x_steps / 2);
		const auto top = grid_center_.y - (steps_.y_step * steps_.y_steps / 2);
		const auto right = grid_center_.x + (steps_.x_step * steps_.x_steps / 2);
		const auto bot = grid_center_.y + (steps_.y_step * steps_.y_steps / 2);
		for (auto x = 0; x <= steps_.x_steps; ++x)
		{
			const auto value = left + x * steps_.x_step;
			lines.append(QLineF(value, top, value, bot));
		}
		for (auto y = 0; y <= steps_.y_steps; ++y)
		{
			const auto value = top + y * steps_.y_step;
			lines.append(QLineF(left, value, right, value));
		}
		painter->drawLines(lines);
	}
	painter->restore();
}


// what does this function do
// void roi_item_serializable::append_capture_items(const int time, const int fov, std::vector<capture_item>& items)
// {
// 	const auto from_tc = x - columns*column_step / 2 + column_step / 2;
// 	const auto from_tr = y - rows*row_step / 2 + row_step / 2;
// 	const auto n = snake_iterator::count(columns, rows);
// 	for (auto rep = 0; rep < repeats; ++rep)
// 	{
// 		for (auto channel_idx : channels)
// 		{
// 			for (auto i = 0; i < n; i++)
// 			{
// 				const auto snake = snake_iterator::iterate(i, columns);
// 				const auto c = snake.column;
// 				const auto r = snake.row;
// 				const auto x_position = from_tc + c*column_step;
// 				const auto y_position = from_tr + r*row_step;
// 				for (auto page_idx = -pages; page_idx <= pages; ++page_idx)
// 				{
// 					//const auto z_position = z + page_idx*page_step;	//Need z position here?
// 					const auto z_position = page_idx*page_step;
// 					const scope_location_xyz loc(x_position, y_position, z_position);
// 					const auto page_label = page_idx + pages;
// 					const roi_name roi_name(fov, time, rep, c, r, page_label);
// 					//const auto move_delay = (page_label == 0) ? stage_move_delay : ms_to_chrono(0);
// 					const auto delays = scope_delays(ms_to_chrono(0));
// 					const auto pos = capture_item(roi_name, delays, loc, channel_idx, false, scope_action::capture);
// 					items.push_back(pos);
// 				}
// 			}
// 		}
// 	}
// }



// void roi_item::set_steps(const grid_steps& steps)
// {
// 	set_grid_steps(steps);
// 	update_grid();
// }



// void roi_item::update_grid(const grid_steps& steps)
// {
// 	const auto old_rect = boundingRect();
//
// 	const auto center = get_grid_center();
// 	xyz_model_ptr->update_four_points(steps, center);
// 	set_grid_steps(steps);
// 	set_grid_center(center);
//
// 	update();
// 	update(old_rect);
// }
//
// void roi_item::update_grid(const scope_location_xy& center)
// {
// 	const auto old_rect = boundingRect();
//
// 	const auto steps = get_roi_steps();  //roi or grid?
// 	xyz_model_ptr->update_four_points(steps, center);
// 	set_grid_center(center);
// 	set_grid_steps(steps);
//
// 	update();
// 	update(old_rect);
// }

