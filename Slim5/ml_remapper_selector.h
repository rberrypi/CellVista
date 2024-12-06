#pragma once
#ifndef ML_REMAPPER_SELECTOR_H
#define ML_REMAPPER_SELECTOR_H

#include "ml_structs.h"

#include <QWidget>

namespace Ui {
	class ml_remapper_selector;
}

class ml_remapper_selector final : public QWidget
{
	Q_OBJECT

		std::unique_ptr<Ui::ml_remapper_selector> ui;

	void update_ml_remapper_selector();

public:

	explicit ml_remapper_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] ml_remapper get_ml_remapper() const;

public slots:
	void set_ml_remapper(const ml_remapper& remapper) const;

signals:
	void ml_remapper_changed(const ml_remapper& remapper);

};

#endif 
