#pragma once
#ifndef CAMERA_CONFIG_SELECTOR_H
#define CAMERA_CONFIG_SELECTOR_H
#include <QWidget>
#include "camera_config.h"
namespace Ui
{
	class camera_aoi_selector;
}

class camera_config_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::camera_aoi_selector> ui_;

	void camera_config_update();
	void update_modes();
public:

	explicit camera_config_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] camera_config get_camera_config() const;

public slots:
	void set_camera_config(const camera_config& mode) const;

signals:
	void camera_config_changed(const camera_config& mode);

};

#endif 
