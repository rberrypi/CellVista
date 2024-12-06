#pragma once
#ifndef COMPACT_LIGHT_PATH_SELECTOR_H
#define COMPACT_LIGHT_PATH_SELECTOR_H
#include "compact_light_path.h"
#include <QGroupBox>
// ReSharper disable once CppInconsistentNaming
class QDoubleSpinBox;
namespace Ui
{
	class compact_light_path_selector;
}

class compact_light_path_selector final : public QGroupBox
{
	Q_OBJECT

	void setup_custom_name();
	void fixup_custom_name();
	std::unique_ptr<Ui::compact_light_path_selector> ui_;

public:

	explicit compact_light_path_selector(QWidget* parent = Q_NULLPTR);
	virtual ~compact_light_path_selector();
	[[nodiscard]] compact_light_path get_compact_light_path() const;

public slots:
	void set_compact_light_path(const compact_light_path& light_path) const;
	void enable_buttons(bool enable) const;
	void set_id(int id);

private:
	int id_;
	void update_light_path();

signals:
	void compact_light_path_changed(const compact_light_path& settings);
	void camera_config_changed(const camera_config& camera_config_settings);
	void remove_me(int id);
};

#endif