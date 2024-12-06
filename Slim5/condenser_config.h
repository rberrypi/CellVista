#pragma once
#ifndef CONDENSER_CONFIG_H
#define CONDENSER_CONFIG_H
#include <QWidget>

#include "instrument_configuration.h"
namespace Ui
{
	class condenser_config;
}
// ReSharper disable once CppInconsistentNaming
class QDoubleSpinBox;
class condenser_config final : public QWidget
{
	Q_OBJECT
		std::unique_ptr<Ui::condenser_config> ui_;

	void update_condenser_position();
public:
	explicit condenser_config(QWidget* parent = Q_NULLPTR);

	[[nodiscard]] condenser_position get_condenser_position() const;

public slots:
	void set_condenser_position(const condenser_position& settings) const;
	void enable_buttons(bool enable) const;

signals:
	void condenser_settings_changed(const condenser_position& settings);
};
#endif