#include "stdafx.h"
#include <iostream>
void fix_windows_console_selection()
{
	//https://blogs.msdn.microsoft.com/oldnewthing/20130506-00/?p=4453
	//Quick edit mode means selecting text will pause the software, which is insane and bad
	auto* h_console = GetStdHandle(STD_INPUT_HANDLE);
	DWORD mode;
	if (!GetConsoleMode(h_console, &mode))
	{
		std::cout << "Failed to get console handle" << std::endl;
	}
	mode &= ~ENABLE_QUICK_EDIT_MODE;
	if (!SetConsoleMode(h_console, mode | ENABLE_EXTENDED_FLAGS))
	{
		std::cout << "Failed to set console mode, did we get the right handle?" << std::endl;
	}
}