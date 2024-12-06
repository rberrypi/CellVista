#pragma once
#ifndef QPHASE_RETRIEVAL_PROCESSING_MODE_SELECTOR_H
#define QPHASE_RETRIEVAL_PROCESSING_MODE_SELECTOR_H
#include <QWidget>
#include "phase_processing.h"

class QComboBox;

namespace Ui {
	class processing_quad_selector;
}
class processing_quad_selector final : public QWidget
{
	Q_OBJECT

	static void hidden_if_empty(QComboBox* cmb);
	static void set_if_found(QComboBox* cmb, const QVariant& data);
	void assert_valid_indexes() const;
	void update_processing_quad();
	std::unique_ptr<Ui::processing_quad_selector> ui;
	void update_retrieval();
	void update_processing_and_denoise();
	[[nodiscard]] phase_processing get_processing() const;
	[[nodiscard]] phase_retrieval get_retrieval() const;
	[[nodiscard]] demosaic_mode get_demosaic() const;
	[[nodiscard]] denoise_mode get_denoise() const;
public:

	explicit processing_quad_selector(QWidget* parent = Q_NULLPTR);
	//virtual ~processing_quad_selector() = default;
	[[nodiscard]] processing_quad get_quad() const;

public slots:
	void set_processing(const processing_quad& quad) const;
	void toggle_everything_but_camera(bool enable) const;
	void switch_layout(bool to_grid);
signals:
	void processing_quad_changed(const processing_quad& quad) const;

};

#endif // QPHASE_RETRIEVAL_PROCESSING_MODE_SELECTOR_H
