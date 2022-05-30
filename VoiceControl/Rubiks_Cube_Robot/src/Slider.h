#ifndef SLIDER_H
#define SLIDER_H

#pragma once

#include <Arduino.h>
#include "Stepper.h"

class Slider:public Stepper
{
public:
	Slider(int step_pin, int dir_pin, int en_pin, int steps_per_rotation, int speed, int max_angle);
	void set_direction(dir dir);

private:
	int max_angle_;	//defines the angle to move grippers away from cube

};

#endif