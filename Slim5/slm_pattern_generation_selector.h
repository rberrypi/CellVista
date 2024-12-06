#pragma once
#ifndef SLM_PATTERN_GENERATION_SELECTOR_H
#define SLM_PATTERN_GENERATION_SELECTOR_H
#include "settings_file.h"
#include <QWidget>
namespace Ui
{
	class slm_pattern_generation_selector;
}

class slm_pattern_generation_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::slm_pattern_generation_selector> ui_;
	void update_slm_pattern_generation();
public:
	explicit slm_pattern_generation_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] slm_pattern_generation get_slm_pattern_generation() const;

public slots:
	void set_slm_pattern_generation(const slm_pattern_generation& settings) const;
	void set_slm_pattern_generation_silent(const slm_pattern_generation& settings);
	void set_processing_double(const processing_double& processing);


signals:
	void slm_pattern_generation_changed(const slm_pattern_generation& settings);
};


#endif