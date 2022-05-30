#ifndef COMMUNICATION_HANDLER_H
#define COMMUNICATION_HANDLER_H

#pragma once
#include <Arduino.h>
#include <ctype.h>	//for isDigit() and isAlpha() functions


class communication_handler
{
public:
	communication_handler();
	bool read_command();
	char get_cmd();
	int get_indicator();
	void init_communication(unsigned long baudrate);

private:
	char command_;
	int indicator_;
	char buffer_[2];
	unsigned long starting_time_;
	void flush_buffer();
	bool validate_cmd(char command, int indicator);
	int char_to_int(char x);
};

#endif