#include <Arduino.h>
#include "Gripper.h"

Gripper::Gripper(int step_pin, int dir_pin, int en_pin, int steps_per_rotation, int speed) : Stepper(step_pin, dir_pin, en_pin, steps_per_rotation, speed)
{
}

void Gripper::set_turns(int quarter_turns, dir dir)
{
	set_angle(90 * quarter_turns, dir);
}
