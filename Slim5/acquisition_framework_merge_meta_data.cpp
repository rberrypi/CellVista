#include "stdafx.h"
#include "acquisition_framework.h"
#include <QDirIterator>
#include <QTextStream>
#include "io_work.h"

void acquisition_framework::merge_meta_data_cs_vs(const std::string& output_directory)
{
	const auto dir = QString::fromStdString(output_directory);
	const auto merged_path = QDir(dir).filePath(QString("Merged.csv"));
	QFile merged_file(merged_path);
	merged_file.open(QFile::WriteOnly);
	QTextStream txt(&merged_file);
	QDirIterator it(dir, QStringList() << "*.csv", QDir::Files, QDirIterator::Subdirectories);
	auto very_first = true;//terrible
	while (it.hasNext())
	{
		auto path = it.next();
		const auto path_as_std_string = path.toStdString();
		auto roi_info = raw_io_work_meta_data::filepath_to_type(path_as_std_string);
		//auto test
		if (roi_info.success)
		{
			auto& roi = roi_info.name;
			QFile csv_in(path);
			csv_in.open(QIODevice::ReadOnly | QIODevice::Text);
			QTextStream in(&csv_in);
			auto first = true;
			const auto prefix = QString::asprintf("%d,%d,%d,%d,%d,%d,%d,", roi.roi, roi.time, roi.repeat, roi.row, roi.column, roi.page, roi_info.channel_route_index);
			while (!in.atEnd())
			{
				const auto line = in.readLine();
				if (very_first)
				{
					very_first = false;
					txt << "roi,time,repeat,row,column,page,channel," << line << Qt::endl;
				}
				if (!first)
				{
					txt << prefix << line << Qt::endl;
				}
				first = false;
			}
			csv_in.close();
		}
	}
}
