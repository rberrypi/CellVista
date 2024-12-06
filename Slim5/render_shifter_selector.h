#pragma once
#ifndef RENDER_SHIFTER_SELECTOR_H
#define RENDER_SHIFTER_SELECTOR_H

#include "render_settings.h"
#include <QWidget>

namespace Ui {
	class render_shifter_selector;
}

class render_shifter_selector final : public QWidget
{
	Q_OBJECT

		std::unique_ptr<Ui::render_shifter_selector> ui;

	void update_render_shifter_selector();

public:

	explicit render_shifter_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] render_shifter get_render_shifter() const;

public slots:
	void set_render_shifter(const render_shifter& remapper) const;

signals:
	void ml_render_shifter(const render_shifter& remapper);

};

#endif 
