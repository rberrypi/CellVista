#pragma once
#ifndef COMPACT_LIGHT_PATH_HOLDER_H
#define COMPACT_LIGHT_PATH_HOLDER_H
#include <QWidget>
#include "compact_light_path.h"
class compact_light_path_selector;
class compact_light_path_holder final : public QWidget
{
	Q_OBJECT

public:
	explicit compact_light_path_holder(QWidget* parent = Q_NULLPTR);
	~compact_light_path_holder() = default;
	[[nodiscard]] std::vector<compact_light_path> get_compact_light_paths() const;
	[[nodiscard]] camera_config get_default_camera_config() const;
	[[nodiscard]] int get_number_channels() const;
	void add_channel(const compact_light_path& new_light_path);

public slots:
	void set_light_paths(const std::vector<compact_light_path>& light_paths);
	void add_one();
	void remove_one(int idx);

signals:
	void channel_removed(int idx);
	void channel_added(int idx);
	void value_changed(const std::vector<compact_light_path>& mode) const;

private:
	std::vector<compact_light_path_selector*> light_path_widgets;
	void reindex_widgets();
	void update_values() const;

};

#endif 