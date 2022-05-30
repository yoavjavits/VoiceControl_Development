#include <Arduino.h>
#include "Slider.h"

Slider::Slider(int step_pin, int dir_pin, int en_pin, int steps_per_rotation, int speed, int max_angle) : Stepper(step_pin, dir_pin, en_pin, steps_per_rotation, speed),
																										  max_angle_{max_angle}
{
}

void Slider::set_direction(dir dir)
{
	set_angle(max_angle_, dir);
}
