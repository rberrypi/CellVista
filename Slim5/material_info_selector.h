#pragma once
#ifndef MATERIAL_INFO_SELECTOR_H
#define MATERIAL_INFO_SELECTOR_H
#include <QWidget>
#include "compute_and_scope_state.h"

namespace Ui
{
	class material_info_selector;
}
class material_info_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::material_info_selector> ui_;	
	void update_material_info();
public:
	explicit material_info_selector(QWidget *parent = Q_NULLPTR);
	[[nodiscard]] material_info get_material_info() const;

public slots:
	void set_material_info(const material_info& settings) const;

signals:
	void material_settings_changed(const material_info& settings);
};


#endif