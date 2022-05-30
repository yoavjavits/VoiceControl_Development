#include "Stepper.h"
#include <Arduino.h>

Stepper::Stepper(int step_pin, int dir_pin, int en_pin, int steps_per_rotation, int speed):
	step_pin_{step_pin},
	dir_pin_{dir_pin},
	en_pin_{en_pin},
	speed_{speed}, 
	steps_per_rotation_{steps_per_rotation},
	steps_to_go_{0},
	next_call_{0}
{
	pinMode(step_pin_, OUTPUT);
	pinMode(dir_pin_, OUTPUT);
	pinMode(en_pin_,OUTPUT);
	digitalWrite(step_pin_, LOW);
	digitalWrite(dir_pin_, LOW);
	digitalWrite(en_pin_,HIGH); //disables stepper
}


bool Stepper::update(unsigned long time)
{
	int delay = 1000/speed_/2; //delay in ms in order to achieve pwm with 50% duty cycle and the desired steps/s
	
	if(!steps_to_go_) 
	{
		return true; //done moving
	}
	else
	{
		if(time >= next_call_)
		{
			
			digitalWrite(step_pin_, !digitalRead(step_pin_));
			next_call_ = time + delay;
	
			if(!digitalRead(step_pin_))
			{
				//Serial.println(steps_to_go_); 
				steps_to_go_--;	//to avoid reducing the counter twice each step this is only executed with the high pulse
			}
				
		}
		return false;
	}
}

void Stepper::set_angle(int angle, dir dir)
{
	
	digitalWrite(dir_pin_, int(dir));		//setting direction
	steps_to_go_ = angle_to_steps(angle);	//calculating requiered #of steps to reach desired angle
}

void Stepper::set_speed(int speed)
{
	speed_ = speed;
}



void Stepper::enable(bool en)
{
	digitalWrite(en_pin_,!en);
}

void Stepper::adjust(dir dir)
{
	digitalWrite(dir_pin_, (int)dir);

	digitalWrite(step_pin_, HIGH);
	delay(20);
	digitalWrite(step_pin_,LOW);
	delay(20);
}

int Stepper::angle_to_steps(int angle)
{
	return angle * ((float)steps_per_rotation_/360);		//returns # of steps //TODO C++ style cast
}
