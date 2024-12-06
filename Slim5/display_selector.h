#pragma once
#ifndef Q_DISPLAY_SELECTOR_H
#define Q_DISPLAY_SELECTOR_H
#include <QWidget>
#include "display_settings.h"

namespace Ui
{
	class display_selector;
}
// ReSharper disable once CppInconsistentNaming
class QDoubleSpinBox;


class display_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::display_selector> ui_;
	void update_display_settings();
	
public:
	explicit display_selector(QWidget* parent = Q_NULLPTR);

	[[nodiscard]] display_settings get_display_settings() const;

public slots:
	void set_display_settings(const display_settings& settings);

private:
	typedef std::vector<QDoubleSpinBox*> box_holder;
	box_holder min_boxes_, max_boxes_;
signals:
	void display_settings_changed(const display_settings& settings);
};


#endif