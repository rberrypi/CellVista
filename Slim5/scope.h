#pragma once
#ifndef SCOPE_H
#define SCOPE_H

#include <QObject>
#include <QStringList>
#include <QRectF>
//
#include <functional>
#include <boost/noncopyable.hpp>
#include "instrument_configuration.h"

// ReSharper disable once CppInconsistentNaming
struct scope_limit_xy final
{
	QRectF xy;
	bool valid;
	explicit scope_limit_xy(const QRectF& xy, const bool valid = true) noexcept : xy(xy), valid(valid) {
	}
	scope_limit_xy() noexcept :scope_limit_xy(QRectF(), false) {}
	[[nodiscard]] bool point_inside_range(const scope_location_xy& pos) const;
};

struct scope_xy_speed final
{
	double xs, xa, ys, ya;
	scope_xy_speed(const double xs, const double xa, const double ys, const double ya) noexcept : xs(xs), xa(xa), ys(ys), ya(ya)
	{

	}
};

class scope_xy_drive : boost::noncopyable
{
protected:
	scope_location_xy xy_current_;
	virtual void move_to_xy_internal(const scope_location_xy& xy) = 0;
	virtual scope_location_xy get_position_xy_internal() = 0;
	scope_limit_xy xy_limits_;
	virtual scope_limit_xy get_stage_xy_limits_internal() = 0;//maybe one day fix so that it doesn't call the internal every time?
public:
	[[nodiscard]] scope_limit_xy get_stage_limits() const
	{
		return xy_limits_;
	}
	scope_location_xy get_position_xy(const bool refresh = false)
	{
		return refresh ? (xy_current_ = get_position_xy_internal()) : xy_current_;
	}
	void move_to_xy(const scope_location_xy& xy);
	virtual ~scope_xy_drive() = default;
	virtual void print_settings(std::ostream&) {}
	void common_post_constructor();	//todo convert to factory pattern;
	[[nodiscard]] bool point_inside_range(const scope_location_xy& pos) const
	{
		return xy_limits_.point_inside_range(pos);
	}
};

struct scope_limit_z final
{
	double zee_min, zee_max;
	bool valid;
	scope_limit_z(const double zee_min, const double zee_max, const bool valid = true) noexcept : zee_min(zee_min), zee_max(zee_max), valid(valid)
	{

	}
	scope_limit_z() noexcept :scope_limit_z(0, 0, false) {}

	[[nodiscard]] bool point_inside_range(const float point) const
	{
		if (!valid)
		{
			return true;
		}

		const auto approx_equals = [](const double value1, const double value2, const double epsilon = 0.001)
		{
			return abs(value1 - value2) < epsilon;
		};

		return (point >= zee_min && point <= zee_max) || approx_equals(point, zee_max);
	}
};

class scope_z_drive : boost::noncopyable
{
public:
	enum class focus_system_status { off, settled, moving };

	scope_z_drive() noexcept : z_current_(scope_location_xy::null()), has_focus_system_(false), focus_system_status_(focus_system_status::off)
	{

	}

	[[nodiscard]] bool has_focus_system() const noexcept
	{
		return has_focus_system_;
	}

	focus_system_status get_focus_system_status(const bool refresh)
	{
		return refresh ? (focus_system_status_ = get_focus_system_internal()) : focus_system_status_;
	}

	float get_position_z(const bool refresh = false)
	{
		return refresh ? (z_current_ = get_position_z_internal()) : z_current_;
	}

	void move_to_z(float z);

	virtual ~scope_z_drive() = default;
	virtual void print_settings(std::ostream&) noexcept {}

	static auto z_move_time_guess() noexcept
	{
		return ms_to_chrono(110);
	}

	virtual scope_limit_z get_z_drive_limits_internal() = 0;

	[[nodiscard]] bool point_inside_range(const float& pos) const
	{
		return zee_limits_.point_inside_range(pos);
	}
protected:
	void common_post_constructor();
	scope_limit_z zee_limits_;
	float z_current_;
	virtual void move_to_z_internal(float z) = 0;
	virtual float get_position_z_internal() = 0;
	bool has_focus_system_;
	focus_system_status focus_system_status_;
	virtual focus_system_status get_focus_system_internal() { return focus_system_status::off; }
};

struct scope_channel_drive_settings
{
	int phase_channel_alias;
	bool is_transmission;
	std::chrono::microseconds channel_off_threshold;
	const static int invalid_phase_channel = -1, off_channel_idx = 0, phase_channel_idx = 1;//hack hack hack
	scope_channel_drive_settings(const int phase_channel_alias, const bool is_transmission, const std::chrono::microseconds& channel_off_threshold) noexcept : phase_channel_alias(phase_channel_alias), is_transmission(is_transmission), channel_off_threshold(channel_off_threshold) {}
	scope_channel_drive_settings() noexcept : scope_channel_drive_settings(invalid_phase_channel, true, ms_to_chrono(4000)) {}
	void load_settings();
	void save_settings();
	static const std::string settings_name;
};

struct condenser_nac_limits final
{
	float nac_min, nac_max;
	bool valid;
	condenser_nac_limits(const float nac_min, const float nac_max, const bool valid = true) noexcept : nac_min(nac_min), nac_max(nac_max), valid(valid)
	{

	}
	condenser_nac_limits() noexcept :condenser_nac_limits(0, 0, false) {}

	[[nodiscard]] bool point_inside_range(const float point) const  noexcept   //TODO Changed to double instead of float
	{
		if (!valid)
		{
			return true;
		}
		return point >= nac_min && point <= nac_max;
	}
};

class scope_channel_drive : public scope_channel_drive_settings
{
	//light path functionality is broken should be merged so that 
	// microscope_light_path get_light_path_internal() works
protected:
	microscope_light_path current_light_path_;
	virtual void move_to_channel_internal(int channel_idx) = 0;
	virtual void move_to_light_path_internal(int light_path_idx) = 0;
	virtual int get_channel_internal() = 0;
	virtual int get_light_path_internal() = 0;
	virtual void move_condenser_internal(const condenser_position& position) = 0;
	virtual condenser_position get_condenser_internal() = 0;
	virtual condenser_nac_limits get_condenser_na_limit_internal() = 0;
	void common_post_constructor();
	condenser_nac_limits nac_limits_;

	[[nodiscard]] bool nac_inside_limits(const double& nac) const  //TODO Changed to double instead of float
	{
		return nac_limits_.point_inside_range(nac);
	}
public:
	bool has_nac;
	bool has_light_path;
	void move_condenser(const condenser_position& position);
	void move_to_channel(int channel_idx);
	void move_to_light_path(int light_path_idx);
	virtual condenser_nac_limits get_condenser_na_limit()
	{
		static auto limit = get_condenser_na_limit_internal();
		return limit;
	}

	condenser_position get_condenser(const bool refresh = false)
	{
		return refresh ? (static_cast<condenser_position&>(current_light_path_) = get_condenser_internal()) : current_light_path_;
	}
	int get_channel(const bool refresh = false)
	{
		return refresh ? (current_light_path_.scope_channel = get_channel_internal()) : current_light_path_.scope_channel;
	}

	int get_light_path(const bool refresh = false)
	{
		if (has_light_path)
		{
			return refresh ? (current_light_path_.light_path = get_light_path_internal()) : current_light_path_.light_path;
		}
		return 0;
	}

	virtual void toggle_lights(bool enable) = 0;
	const static std::string channel_off_str;
	const static std::string channel_phase_str;
	std::vector<std::string> light_path_names;//shoudl combine with has_light_path(!)
	std::vector<std::string> channel_names;
	std::vector<std::string> condenser_names;
	scope_channel_drive() noexcept : has_nac(false), has_light_path(false)
	{
		load_settings();
	}
	virtual ~scope_channel_drive()
	{
		save_settings();
	}
	virtual void print_settings(std::ostream&) noexcept {}
	//
};

class microscope final : public QObject
{
	Q_OBJECT

		Q_DISABLE_COPY(microscope)

protected:
	//
	typedef boost::container::static_vector<std::function<void()>, 5> scope_async_actions;///wait, maybe the llvm thing?
	static void static_async_launcher(const scope_async_actions& input, bool async);
	void wait_for_pfs(bool async, bool stage_moving) const;
	const char* unused = "";
public:
	explicit microscope(QObject* parent);
	virtual ~microscope();
	//
	void print_settings(std::ostream& input) const;
	static std::unique_ptr<scope_channel_drive> get_scope_channel_drive();
	static std::unique_ptr<scope_xy_drive> get_scope_xy_drive();
	static std::unique_ptr<scope_z_drive> get_microscope_z_drive();
	std::unique_ptr<scope_channel_drive> chan_drive;
	std::unique_ptr<scope_xy_drive> xy_drive;
	std::unique_ptr<scope_z_drive> z_drive;
	void move_to(const microscope_move_action& pos, bool async);
	void move_to(const scope_location_xyz& pos, bool async);
	void move_light_path(const microscope_light_path& light_path, bool async);
	[[nodiscard]] QStringList get_channel_settings_names() const;//helper function
	[[nodiscard]] QStringList get_light_path_names() const;//helper function
	[[nodiscard]] QStringList get_condenser_settings_names() const;//helper function
	[[nodiscard]] microscope_state get_state(bool refresh = false) const;
	//
	static const std::chrono::microseconds shutter_time;
signals:
	void focus_system_engaged(bool engaged);
};

#endif