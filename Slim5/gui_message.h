#pragma once
#ifndef GUI_MESSAGE_H
#define GUI_MESSAGE_H
#include <QVariant>
#include <mutex>
#include <queue>
#include "io_work.h"

struct gui_message final
{
	gui_message_kind kind;
	QVariant val;
	gui_message() noexcept: kind(gui_message_kind::none)
	{

	}
	gui_message(const gui_message_kind kind, QVariant  val) noexcept:
		kind(kind), val(std::move(val))
	{

	}
};

class gui_messages
{
public:
	std::map<QString, int> setting;
	void push_live_message(const gui_message& msg)
	{
		std::unique_lock<std::mutex> lk(gui_messages_queue_m_);
		gui_messages_queue_.push(msg);
	}
	gui_message pop_live_message()
	{
		std::unique_lock<std::mutex> lk(gui_messages_queue_m_);
		if (gui_messages_queue_.empty())
		{
			return gui_message();
		}
		auto msg = gui_messages_queue_.front();
		gui_messages_queue_.pop();
		return msg;
	}
private:
	std::mutex gui_messages_queue_m_;
	std::queue<gui_message> gui_messages_queue_;
};

#endif