#pragma once
#ifndef SLM_PATTERN_MODEL_H
#define SLM_PATTERN_MODEL_H
#include "modulator_configuration.h"
#include <QAbstractTableModel>
// ReSharper disable once CppInconsistentNaming
class QXmlStreamWriter;
class per_pattern_modulator_settings_patterns_model final : public QAbstractTableModel
{
	Q_OBJECT
	per_pattern_modulator_settings_patterns patterns;
public:
	//For read only
	explicit per_pattern_modulator_settings_patterns_model(QObject *parent = nullptr);
	int rowCount(const QModelIndex &parent = QModelIndex()) const override;
	int columnCount(const QModelIndex & = QModelIndex()/*parent*/) const override;
	
	//for editing
	QVariant data(const QModelIndex &index, int role) const override;
	bool setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole) override;
	Qt::ItemFlags flags(const QModelIndex & /*index*/) const override;
	
	//for resizing
	bool insertRows(int row, int count, const QModelIndex & parent = QModelIndex()) override;
	bool insertColumns(int column, int count, const QModelIndex & parent = QModelIndex()) override;
	bool removeRows(int row, int count, const QModelIndex & parent = QModelIndex()) override;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
	//
	void set_per_pattern_modulator_settings_patterns(const per_pattern_modulator_settings_patterns& patterns);
	const per_pattern_modulator_settings_patterns& get_pattern_modulator_settings_patterns() const;
	static bool is_visible( int idx,  slm_mode mode, bool is_color);
};
#endif