#pragma once
#ifndef FULL_INTERFACE_GUI_H
#define FULL_INTERFACE_GUI_H

#include <QMainWindow>

#include "acquisition.h"
#include "cgal_triangulator.h"
#include "compact_light_path.h"
#include "roi_model.h"
#include "settings_file.h"

enum class capture_mode;
class slim_four;
struct calibration_info;
struct trakem2_stage_coordinate_to_pixel_mapper;
struct acquisition;
class trakem2_stitching_dialog;
class trakem2_processors;
struct calibration_attributes;
struct full_interface_gui_settings
{
	typedef std::vector<compact_light_path> light_path_settings;
	light_path_settings light_paths;
	int cmb_acquire_modes;
	int full_iteration_times;
	bool interpolate_roi_enabled;
	bool interpolate_roi_global_enabled;
	std::string meta_data;
	channel_switching_order switch_channel_mode;
	filename_grouping_mode filename_grouping;

	full_interface_gui_settings() noexcept : full_interface_gui_settings(light_path_settings(), 0, 0, false, false, std::string(), channel_switching_order::switch_channel_per_roi, filename_grouping_mode::same_folder) {};

	full_interface_gui_settings(const light_path_settings& light_paths, const int acquire_modes, const int full_iteration_times, const bool interpolate_roi_enabled, const bool interpolate_roi_global_enabled, const std::string& meta_data, const channel_switching_order switch_channel_mode, const filename_grouping_mode mode) : light_paths(light_paths), cmb_acquire_modes(acquire_modes), full_iteration_times(full_iteration_times), interpolate_roi_enabled(interpolate_roi_enabled), interpolate_roi_global_enabled(interpolate_roi_global_enabled), meta_data(meta_data), switch_channel_mode(switch_channel_mode), filename_grouping(mode) {};

	[[nodiscard]] bool item_approx_equals(const full_interface_gui_settings& settings) const
	{
		const auto comparison_functor = [](const compact_light_path& a, const compact_light_path& b)
		{
			return a.item_approx_equals(b);
		};
		const auto check_light_path_vector = std::equal(light_paths.begin(), light_paths.end(), light_paths.begin(), comparison_functor);

		return check_light_path_vector && cmb_acquire_modes == settings.cmb_acquire_modes && full_iteration_times == settings.full_iteration_times && interpolate_roi_enabled == settings.interpolate_roi_enabled && interpolate_roi_global_enabled == settings.interpolate_roi_global_enabled
			&& meta_data == settings.meta_data && switch_channel_mode == settings.switch_channel_mode && filename_grouping == settings.filename_grouping;
	}

};

namespace Ui {
	class full_interface_gui;
}

class full_interface_gui final : public QMainWindow
{
	Q_OBJECT

	static std::array<QString, 3> scan_state_labels;
	typedef std::unordered_map<channel_switching_order, const QString> roi_switching_map;
	const static roi_switching_map roi_switching_settings;
	const static std::string default_scan_settings_name;

	QGraphicsEllipseItem* pointer_;
	std::unique_ptr<roi_model> rois_;
	std::unique_ptr<QGraphicsScene> gfx_;
	trakem2_stitching_dialog* trakem2_dialog_;

	settings_file settings_file_;
	
	void setup_disk_size();
	void setup_graphics_surface();
	void setup_buttons();
	void setup_rois_table();
	[[nodiscard]] frame_size default_sensor_size_in_pixels() const;
	[[nodiscard]] roi_item_serializable get_default_item() const;
	[[nodiscard]] int  get_roi_index() const;
	void select_roi(int item_idx) const;
	void setup_acquire_buttons();
	void verify_acquire_button();
	void setup_file_grouping_modes();
	void fill_column_from_selection() const;
	void update_model_selection(int selected_roi_idx) const;

	//void setup_step_size() const;
	void setup_navigation_buttons();
	void setup_channels() const;
	void setup_common_channels();
	void add_common_channel(int button_idx) const;
	//QString get_workspace_path() const;

	void setup_xyz_model_item(int row) const;
	void update_xyz_model_selection(int selected_xyz_idx) const;
	template <bool IsTop> [[nodsicard]] void peak_top_bottom(bool checked) const;

	[[nodiscard]] scope_location_xyz get_step_sizes(const frame_size& frame) const noexcept;
	
	slim_four* slim_four_handle;
	bool is_valid_acquisition();
protected:
		void closeEvent(QCloseEvent* event) override;

public:
	full_interface_gui(const live_gui_settings& live_gui_settings, const settings_file& settings_file, slim_four* slim_four_handle, QMainWindow* parent);
	std::unique_ptr<Ui::full_interface_gui> ui_;

	[[nodiscard]] std::vector<channel_settings> get_channel_settings() const;		
	void set_live_gui_settings(const live_gui_settings& settings);							

	[[nodiscard]] full_interface_gui_settings get_saveable_settings() const;
	void set_saveable_settings(const full_interface_gui_settings& settings) const;

	[[nodiscard]] std::vector<float> get_z_for_whole_roi(int roi_idx) const;
	void set_zee_for_whole_roi(float zee, int roi_idx) const;
	void set_zee_for_whole_roi(const std::vector<float>& z_values, int roi_idx) const;
	void increment_zee_for_whole_roi(float zee, int roi_idx) const;

	void setup_load_save_dialog();
	[[nodiscard]] QString get_dir() const;

	void resizeEvent(QResizeEvent* event) override;

	static QRectF get_camera_size_in_stage_coordinates(const camera_config& settings, const pixel_dimensions& dimensions);
	
public slots:
	void gui_enable(bool enable);
	//
	void set_focus() const;			//XYZ
	void un_verify_focus_points(int roi_row) const;
	//
	void step_grid(bool x_axis, bool inc) const;
	void center_grid() const;
	//void update_step_size() const;
	void fit_roi_in_view() const;
	//
	void save_metadata() const;
	//
	void set_file_settings(const settings_file& settings_file) ;
	void set_microscope_state(const microscope_state& state) ;
	void wrangle_convex_hull() const;
	void set_six_well_plate() const;

	void set_available_bytes(size_t bytes) ;
	void focus_system_engaged(bool enable) ;
	void update_capture_info();
	void update_pointer_frame_size() const;
	void paintEvent(QPaintEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	void insert_tomogram();
	void insert_point() const;
	void remove_point() const;
	[[nodiscard]] int choose_next_roi() const;
	void goto_point(int roi_row = -1) const;
	void set_roi_xyz() const;
	[[nodiscard]] bool set_roi_z(int roi_row = -1) const;
	void set_all_roi_z() const;
	void do_scan();
	[[nodsicard]] bool wrangle_scan() ;
	[[nodsicard]] bool wrangle_capture_items(acquisition& route, std::set<int>& channels_used);
	void save_cereal_file(const QString& path);
	void load_cereal_file(const QString& path);

	void write_trakem2_xml(const trakem2_stage_coordinate_to_pixel_mapper& mapper, const std::string& basedir, const calibration_info& xy);
	void setup_trakem2();
	void trakem2_calibration(const calibration_info& xy);
signals:
	void roi_changed();
	void pixel_ratio_changed(const float& pixel_ratio); 
	void start_acquisition(capture_mode capture_mode);
	void stop_acquisition();
};


#endif
