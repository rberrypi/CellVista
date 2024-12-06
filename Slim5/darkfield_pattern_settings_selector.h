#pragma once
#ifndef DARKFIELD_PATTERN_SETTINGS_SELECTOR_H
#define DARKFIELD_PATTERN_SETTINGS_SELECTOR_H
#include "modulator_configuration.h"
#include <QWidget>
namespace Ui
{
	class darkfield_pattern_settings_selector;
}

class darkfield_pattern_settings_selector final : public QWidget
{
	Q_OBJECT

	std::unique_ptr<Ui::darkfield_pattern_settings_selector> ui_;
	void update_darkfield_settings();
	void set_complete(bool is_complete);
public:
	explicit darkfield_pattern_settings_selector(QWidget *parent = Q_NULLPTR);
	[[nodiscard]] darkfield_pattern_settings get_darkfield_pattern_settings() const;

public slots:
	void set_darkfield_pattern_settings(const darkfield_pattern_settings& settings) ;
	void set_darkfield_pattern_settings_silent(const darkfield_pattern_settings& settings) ;

	void enable_buttons(bool enable) const;

signals:
	void darkfield_pattern_settings_changed(const darkfield_pattern_settings& settings);
};


#endif