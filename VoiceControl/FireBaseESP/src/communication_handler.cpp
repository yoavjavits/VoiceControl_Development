#include <Arduino.h>
#include "communication_handler.h"

CommunicatioHandler::CommunicatioHandler() : command_{'\n'}, indicator_{0}, buffer_{0}, starting_time_{millis()}
{
}

bool CommunicatioHandler::read_command()
{
	static constexpr int timeout = 1000;

	if (Serial2.available() >= 2) // if 2 bytes are stored in the rx buffer
	{
		for (int i = 0; Serial2.available() && i < 2; i++) // rx reading buffer into locla buffer array
		{
			buffer_[i] = Serial2.read();
		}
		if (validate_cmd(buffer_[0], buffer_[1])) // validating command received
		{
			command_ = buffer_[0];
			indicator_ = char_to_int(buffer_[1]);
			// flush_buffer();
			return true;
		}
		else // invalid command
		{
			//Serial.write("NUL");
			return false;
		}
	}
	else if (millis() - starting_time_ >= timeout)
	{
		flush_buffer();
		starting_time_ = millis();
		return false; // no command received in time
	}

	return false; // not enough bytes received yet
}

char CommunicatioHandler::get_cmd()
{
	return command_;
}

int CommunicatioHandler::get_indicator()
{
	return indicator_;
}

/*void CommunicatioHandler::init_communication()
{
	mySerial(16,17);
}*/

void CommunicatioHandler::flush_buffer()
{
	while (Serial2.available())
	{
		char temp_buffer = Serial2.read();
	}
}

bool CommunicatioHandler::validate_cmd(char command, int indicator)
{
	return (isAlpha(command) && isDigit(indicator));
}

int CommunicatioHandler::char_to_int(char x)
{
	return static_cast<int>(x) - 48;
}