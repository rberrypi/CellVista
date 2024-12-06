#include "stdafx.h"
#include "slm_control.h"
#include "device_factory.h"
#include "image_j_pipe.h"
#include <QMessageBox>
#include <QDir>
#include <QFile>
#include <QTextStream>

void slm_control::setup_image_j_interop() const
{
	static auto disconnected_icon = QPixmap(":/images/checkbox-disconnected.svg");
	static auto connected_icon = QPixmap(":/images/checkbox-connected.svg");
	const QFile path(QString::fromStdString(D->ij->imagej_exe_path));
	QFileInfo file_info(path);
	const auto ij_dir = file_info.absoluteDir().absolutePath();
	ui_.txtImageJFolder->setText(ij_dir);
	connect(ui_.txtImageJFolder, &folder_line_edit::textChanged, this, &slm_control::validate_image_j_folder);
	connect(ui_.btnSaveFiles, &QPushButton::pressed, [] {D->ij->save_pipe_path(); });
}

void slm_control::validate_image_j_folder(const QString& imagej_folder_path) const
{
	const auto error_message = [](const QString& text) { QMessageBox msg(QMessageBox::Icon::Warning, tr("Oh no"), text, QMessageBox::Abort); };
	auto normalized_dir = QDir(imagej_folder_path);
	const auto check_asset = [normalized_dir](const QString& asset_path)
	{
		auto path = normalized_dir.filePath(asset_path);
		return std::make_pair(QFile(path).exists(),path);
	};
	const auto has_fiji_imagej = check_asset(tr("ImageJ-win64.exe"));
	if (has_fiji_imagej.first)
	{
		//installed into /AutoRun folder
		if (check_asset(tr("macros/AutoRun")).first)
		{
			const auto has_auto_run_macro = check_asset("macros/AutoRun/auto_start_qpi_receiver.ijm");
			if (!has_auto_run_macro.first)
			{
				
				QFile file(has_auto_run_macro.second);
				if (!file.open(QIODevice::WriteOnly))
				{
					error_message(tr("Can't write autostart file into ImageJ macro directory"));
				}
				else
				{
					const auto msg = "run(\"QPI Receiver\")";
					file.write(msg);
					file.close();
				}
			}
		}
		//Copy plugin
		const auto has_receiver = check_asset(tr("plugins/QPI_Receiver.jar"));
		if (!has_receiver.first)
		{
			QFile local_receiver_file(tr("QPI_Receiver.jar"));
			const auto local_file_exists = local_receiver_file.exists();
			if (local_file_exists)
			{
				if (!QFile::copy(local_receiver_file.fileName(), has_receiver.second))
				{
					error_message(tr("Can't write receiver file"));
				}
			}
		}
		//Copy LUTs
		const auto has_lut_test_lut = check_asset(tr("luts/QPI Gray.lut"));
		if (!has_lut_test_lut.first)
		{
			auto lut_folder = check_asset(tr("luts/")).second;
			render_settings::deploy_luts(lut_folder.toStdString());
		}
		{
			QMessageBox msg_box;
			msg_box.setText("ImageJ Files Installed");
			msg_box.exec();
		}
	}
	D->ij->set_image_j_exe_path(has_fiji_imagej.second);
	
}
