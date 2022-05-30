#include <Arduino.h>
#include "Slider.h"
#include "Gripper.h"
#include "Stepper.h"
#include "Solver.h"
#include "hardware_parameters.h"

Solver qbot;

void setup() 
{
	qbot.init(baudrate);
}

void loop() 
{
	while(!qbot.communication_handle.read_command());
	while(!qbot.execute_comand(qbot.communication_handle.get_cmd(), qbot.communication_handle.get_indicator()));
}
