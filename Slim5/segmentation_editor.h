#pragma once
#ifndef SEGMENTATION_EDITOR_H
#define SEGMENTATION_EDITOR_H
#include "render_settings.h"
#include <QWidget>

namespace Ui {
	class SegmentationEditor;
}
class segmentation_editor final : public QWidget
{
	Q_OBJECT

	void update_segmentation_settings() ;
public:
	explicit segmentation_editor(const segmentation_settings& settings, QWidget* parent = Q_NULLPTR);
	explicit segmentation_editor(QWidget* parent = Q_NULLPTR) : segmentation_editor(segmentation_settings(), parent) {}
	virtual	~segmentation_editor();

	[[nodiscard]] segmentation_settings get_segmentation() const;

public slots:
	void set_segmentation(const segmentation_settings& settings) const;
	void enable_buttons(bool enable) const;
private:
	std::unique_ptr<Ui::SegmentationEditor> ui_;

signals:
	void segmentation_changed(const segmentation_settings& settings);
};

#endif