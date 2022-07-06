#ifndef COMMUNICATION_HANDLER_H
#define COMMUNICATION_HANDLER_H

#include <ctype.h>

class CommunicatioHandler
{
public:
	CommunicatioHandler();
	bool read_command();
	char get_cmd();
	int get_indicator();

private:
	char command_;
	int indicator_;
	char buffer_[2];
	unsigned long starting_time_;

	void flush_buffer();
	bool validate_cmd(char command, int indicator);
	int char_to_int(char x);
	// void CommunicatioHandler::init_communication();
};

#endif