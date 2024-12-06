#pragma once
#ifndef LIGHT_PATH_SELECTOR_H
#define LIGHT_PATH_SELECTOR_H

#include <QWidget>
#include "instrument_configuration.h"

namespace Ui
{
	class light_path_selector;
}

class light_path_selector final : public QWidget
{
	Q_OBJECT
		std::unique_ptr<Ui::light_path_selector> ui_;

	void update_light_path();
public:

	explicit light_path_selector(QWidget* parent = Q_NULLPTR);
	void set_light_path_selector(const microscope_light_path& settings) const;
	void enable_buttons(bool enable) const;
	[[nodiscard]] microscope_light_path get_light_path() const;
	void set_hidden(bool hidden) const;
	void fixup_light_path(int camera_idx);
signals:
	void light_path_selector_changed(const microscope_light_path& settings);
};

#endif