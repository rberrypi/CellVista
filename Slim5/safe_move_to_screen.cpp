#include "stdafx.h"
#include "safe_move_to_screen.h"
#include <QGuiApplication>
#include <QMainWindow>
#include <QScreen>
void safe_move_to_screen(QMainWindow* window, int screen_idx)
{
	const auto screens = QGuiApplication::screens();
	const auto first_or_second = std::min(screens.size() - 1, 1);
	const auto* screen = screens.at(first_or_second);
	const auto screen_resolution = screen->geometry();
	window->move(QPoint(screen_resolution.x(), screen_resolution.y()));
}