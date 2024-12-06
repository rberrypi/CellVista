#pragma once
#ifndef ROI_ITEM_SHARED_H
#define ROI_ITEM_SHARED_H
#include "approx_equals.h"
#include "capture_item.h"
#include "xyz_focus_points_model.h"
#include "cgal_triangulator.h"

#define ITEM_ID_IDX 0
#define ITEM_X_IDX 1
#define ITEM_Y_IDX 2
#define ITEM_COLS_IDX 3
#define ITEM_ROWS_IDX 4
#define ITEM_PAGES_IDX 5
#define ITEM_STEP_COL_IDX 6
#define ITEM_STEP_ROW_IDX 7
#define ITEM_STEP_PAGE_IDX 8
#define ITEM_DELAYS_IDX 9
#define ITEM_CHANNEL_IDX 10
#define ITEM_REPEATS_IDX 11
#define ITEM_SETS_BG_IDX 12
#define ITEM_SYNC_IDX 13
#define ITEM_VERIFIED_IDX 14

typedef std::vector<std::pair<float, float>> xy_pairs;

struct roi_item_meta_data final
{
	QString label, tooltip;
	bool updates_size;
	int debug_idx;
};

extern roi_item_meta_data roi_model_info[ITEM_VERIFIED_IDX + 1];

struct roi_item_dimensions
{
	int columns, rows, pages;
	float column_step, row_step, page_step;

	[[nodiscard]] int get_items() const noexcept
	{
		return columns * rows * (1 + 2 * pages);
	}
};

struct roi_item_shared : scope_delays
{
	roi_item_shared() noexcept: roi_item_shared(scope_delays(), { 1 }, 0, false, false, false) {}
	roi_item_shared(const scope_delays& delays, const fl_channel_index_list& channels, const int repeats, const bool io_sync_point, const bool sets_bg, const bool grid_selected) noexcept : scope_delays(delays), channels(channels), repeats(repeats), io_sync_point(io_sync_point), sets_bg(sets_bg), grid_selected_(grid_selected) {}

	fl_channel_index_list channels;
	int repeats;
	bool io_sync_point, sets_bg;
	bool grid_selected_;

};

struct roi_item_serializable final : roi_item_shared, scope_location_xy, roi_item_dimensions
{
	roi_item_serializable(const roi_item_shared& shared, const scope_location_xy& location, const roi_item_dimensions& dimensions, const std::vector<scope_location_xyz>& foc_points) noexcept : roi_item_shared(shared), scope_location_xy(location), roi_item_dimensions(dimensions), focus_points(foc_points) {}

	roi_item_serializable() noexcept: roi_item_serializable(roi_item_shared(), scope_location_xy(), roi_item_dimensions(), std::vector<scope_location_xyz>()) {}

	static void merge_b_into_a(roi_item_serializable& a, const roi_item_serializable& b);
	//void append_capture_items(int time, int fov, std::vector<capture_item>& items);

	std::vector<scope_location_xyz> focus_points;

	[[nodiscard]] bool is_load_valid(const roi_item_serializable& item) const
	{
		if (!approx_equals(item.x, x) || !approx_equals(item.y, y) || item.columns != columns || item.rows != rows || item.pages != pages || !approx_equals(item.column_step, column_step) || !approx_equals(item.row_step, row_step) || !approx_equals(item.page_step, page_step) || item.repeats != repeats || item.io_sync_point != io_sync_point || item.sets_bg != sets_bg || !std::equal(channels.begin(), channels.end(), item.channels.begin()))
		{
			return false;
		}
		return true;
	}

};

struct rgb {
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

class roi_item final : public roi_item_shared, roi_item_dimensions, public QAbstractGraphicsShapeItem
{
	int id_;
	grid_steps steps_;
	scope_location_xy grid_center_;
	//bool grid_selected_;
	static bool do_interpolation_local_;
	static bool do_interpolation_global_;

protected:
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;

public slots:
	void set_grid_steps(const grid_steps& steps);
	void set_grid_center(const scope_location_xy& center);

public:
	roi_item(int id, roi_item_serializable& data, QGraphicsItem* parent = nullptr);
	cgal_triangulator triangulator;		//has to be declared before xyz model
	xyz_focus_points_model xyz_model;

	static float zee_max_global;
	static float zee_min_global;

	[[nodiscard]] QRectF boundingRect() const override;
	void grid_selected(bool selected);
	[[nodiscard]] float query_triangulator(const scope_location_xy& point) const;
	//
	void update_grid();
	void set_center(const scope_location_xy& center);

	[[nodiscard]] grid_steps get_roi_steps() const;
	[[nodiscard]] xy_pairs get_xy_focus_points() const;
	void set_xy_focus_points(const xy_pairs& xy_focus_points, float displacement_x, float displacement_y);

	[[nodiscard]] std::vector<float> get_z_all_focus_points() const;
	void set_all_focus_points(float zee_level);
	void set_all_focus_points(const std::vector<float>& z_values);
	void increment_all_focus_points(float increment);
	//
	[[nodiscard]] float get_x() const;
	void set_x(float x);
	[[nodiscard]] float get_y() const;
	void set_y(float y);
	//
	void set_delays(const scope_delays& delays);
	[[nodiscard]] bool focus_points_verified() const;
	void verify_focus_points(bool verify);
	//
	void set_roi_item_serializable(const roi_item_serializable& serializable);
	[[nodiscard]] roi_item_serializable get_roi_item_serializable() const;

	[[nodiscard]] scope_location_xy get_grid_center() const noexcept
	{
		return grid_center_;
	}

	[[nodiscard]] grid_steps get_grid_steps() const noexcept
	{
		return steps_;
	}
	template <bool IsLocal>
	static void enable_interpolation(const bool enable)
	{
		if (IsLocal) {
			do_interpolation_local_ = enable;
		}
		else
		{
			do_interpolation_global_ = enable;
		}
	}
	static bool do_interpolation() noexcept
	{
		return do_interpolation_local_ || do_interpolation_global_;
	}

	[[nodiscard]] int get_id() const noexcept
	{
		return id_;
	}
	void set_id(const int id)
	{
		id_ = id;
		update();
	}

	[[nodiscard]] int get_columns() const noexcept
	{
		return columns;
	}
	void set_columns(const int columns)
	{
		this->columns = columns;
		update_grid();
	}

	[[nodiscard]] int get_rows() const noexcept
	{
		return rows;
	}
	void set_rows(const int rows)
	{
		this->rows = rows;
		update_grid();

	}

	[[nodiscard]] int get_pages() const noexcept
	{
		return pages;
	}
	void set_pages(const int pages)
	{
		this->pages = pages;
		update();
	}

	[[nodiscard]] float get_column_step() const noexcept
	{
		return column_step;
	}
	void set_column_step(const float column_step)
	{
		this->column_step = column_step;
		update_grid();
	}

	[[nodiscard]] float get_row_step() const noexcept
	{
		return row_step;
	}
	void set_row_step(const float row_step)
	{
		this->row_step = row_step;
		update_grid();
	}

	[[nodiscard]] float get_page_step() const noexcept
	{
		return page_step;
	}
	void set_page_step(const float page_step)
	{
		this->page_step = page_step;
		update();
	}
	//
	enum { Type = UserType + 2 };

	[[nodiscard]] int type() const override
	{
		return Type;
	}
};
#endif