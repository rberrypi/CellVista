#pragma once
#ifndef BAND_PASS_FILTER_H
#define BAND_PASS_FILTER_H

#include <QWidget>
#include "compute_and_scope_state.h"

namespace Ui
{
	class band_pass_settings;
}

class band_pass_settings_selector final : public QWidget
{
	Q_OBJECT

		std::unique_ptr<Ui::band_pass_settings> ui_;
	void band_pass_settings_update();
	void set_band_pass_toggle(bool enable) const;
	
public:
	explicit band_pass_settings_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] band_pass_settings get_band_pass_settings() const;

public slots:
	void set_band_pass_settings(const band_pass_settings& band_pass_settings) const;

signals:
	void band_pass_settings_changed(const band_pass_settings& band_pass_settings);
};

#endif
