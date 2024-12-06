#pragma once
#ifndef DISTORTED_DONUT_SELECTOR_H
#define DISTORTED_DONUT_SELECTOR_H
#include <QWidget>
#include "modulator_configuration.h"
namespace Ui
{
	class distorted_donut_selector;
}
class distorted_donut_selector final : public QWidget
{
	Q_OBJECT

	std::unique_ptr<Ui::distorted_donut_selector> ui_;
	void set_complete(bool is_complete);
public:
	explicit distorted_donut_selector(QWidget *parent = Q_NULLPTR);
	[[nodiscard]] distorted_donut get_distorted_donut() const;

public slots:
	void set_distorted_donut_silent(const distorted_donut& distorted_donut);
	void set_distorted_donut(const distorted_donut& distorted_donut);
	void bump_donut(int x, int y, int inner, int outer);

signals:
	void value_changed(const distorted_donut& distorted_donut) const;

private:
	void update_values();
};

#endif 