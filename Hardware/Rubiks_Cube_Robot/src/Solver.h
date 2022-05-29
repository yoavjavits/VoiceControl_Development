#ifndef SOLVER_H
#define SOLVER_H

#pragma once
#include <Arduino.h>
#include "Slider.h"
#include "Gripper.h"
#include "communication_handler.h"
#include "enum.h"
#include "hardware_parameters.h"

class Solver
{
public:
	Solver();
	void init(long baud);	//starts the serial communication, has to be called within the setup() function, starts up motors
	bool execute_comand(char command, int indicator);
	bool enable_steppers();
	bool disable_steppers();

	communication_handler communication_handle;	//as the name implies, it handles the serial communication to the control software

private:
	bool turn_side(Gripper &gripper, dir dir, int turns, unsigned long time);
	bool slide(Slider &slider, dir dir, unsigned long time);
	bool turn_top_bot(dir dir, int turns, cube_sides side, unsigned long time);
	bool adjust_cmd(int indicator, dir dir);
	bool turn_back(Gripper &gripper, Slider &slider, dir direction, int turns, unsigned long time); 
	Slider slider_Y_;
	Slider slider_X_;
	Gripper gripper_L_;
	Gripper gripper_R_;
	Gripper gripper_F_;
	Gripper gripper_B_;
	state moving_state_top_bot_;
	state moving_state_turn_back_;

	
};

#endif