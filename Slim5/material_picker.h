#pragma once
#ifndef MATERIAL_PICKER_H
#define MATERIAL_PICKER_H
#include <QMainWindow>
#include "compute_and_scope_state.h"
#include "phase_processing.h"

struct material_covers final
{
	QString name;
	bool show_lambda;
	bool show_media;
	bool show_obj;
	bool show_increment;
	bool show_microns;
};
namespace Ui
{
	class material_picker;
}

class material_picker final : public QMainWindow
{
	Q_OBJECT

		std::unique_ptr<Ui::material_picker> ui;
	const static std::unordered_map<phase_processing, material_covers> covers;
	const static material_covers unsupported_cover;

	[[nodiscard]] material_info get_material_info() const;
	void material_info_update() const;
public:
	explicit material_picker(const phase_processing& processing, const material_info& info, QWidget* parent);

public slots:
	void set_phase_processing(phase_processing idx) const;
	void set_material_info(const material_info& info) const;
signals:
	void material_info_updated(const material_info& info) const;
	void delete_material_picker() const;

};

#endif