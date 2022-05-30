#ifndef GRIPPER_H
#define GRIPPER_H

#pragma once

#include <Arduino.h>
#include "Stepper.h"


class Gripper: public Stepper
{
public:
	Gripper(int step_pin, int dir_pin,int en_pin, int steps_per_rotation, int speed);
	void set_turns(int quarter_turns, dir dir);
	bool moving;
};

#endif