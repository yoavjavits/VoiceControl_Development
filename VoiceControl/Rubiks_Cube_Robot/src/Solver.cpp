#include "Solver.h"


Solver::Solver():
communication_handle(),
slider_Y_(sliderY_step, sliderY_dir, sliderY_en, steps_per_rot, slider_speed, max_angle),
slider_X_(sliderX_step, sliderX_dir, sliderX_en, steps_per_rot, slider_speed, max_angle),
gripper_L_(gripperL_step, gripperL_dir, gripperL_en, steps_per_rot, gripper_speed),
gripper_R_(gripperR_step, gripperR_dir, gripperR_en, steps_per_rot, gripper_speed),
gripper_F_(gripperF_step, gripperF_dir, gripperF_en, steps_per_rot, gripper_speed),
gripper_B_(gripperB_step, gripperB_dir, gripperB_en, steps_per_rot, gripper_speed),
moving_state_top_bot_{state::unlock1},
moving_state_turn_back_{state::turn}
{

}

void Solver::init(long baud)
{
	//communication_handle.init_communication(baudrate);
	Serial.begin(9600);
  	Serial2.begin(115200);
	enable_steppers();
}


bool Solver::execute_comand(char command, int indicator)
{
	bool executed = false;
	unsigned long current_time = millis(); 
	
	switch(command)
	{
	case 'R':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_R_, slider_X_, dir::cw,indicator, current_time);
		else
			executed = turn_side(gripper_R_,dir::cw,indicator, current_time);
		break;

	case 'r':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_R_, slider_X_, dir::ccw,indicator, current_time);
		else
		executed = turn_side(gripper_R_,dir::ccw,indicator, current_time);
		break;

	case 'L':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_L_, slider_X_, dir::cw,indicator, current_time);
		else
		executed = turn_side(gripper_L_,dir::cw,indicator, current_time);
		break;
	
	case 'l':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_L_, slider_X_, dir::ccw,indicator, current_time);
		else
		executed = turn_side(gripper_L_,dir::ccw,indicator, current_time);
		break;

	case 'F':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_F_, slider_Y_, dir::cw,indicator, current_time);
		else
		executed = turn_side(gripper_F_,dir::cw,indicator, current_time);
		break;

	case 'f':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_F_, slider_Y_, dir::ccw,indicator, current_time);
		else
		executed = turn_side(gripper_F_,dir::ccw,indicator, current_time);
		break;

	case 'B':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_B_, slider_Y_, dir::cw,indicator, current_time);
		else
		executed = turn_side(gripper_B_,dir::cw,indicator, current_time);
		break;
	
	case 'b':
		if(indicator % 2 != 0)	//if odd number of quarter turns
			executed = turn_back(gripper_B_, slider_Y_, dir::ccw,indicator, current_time);
		else
		executed = turn_side(gripper_B_,dir::ccw,indicator, current_time);
		break;
		
	case 'U':
		executed = turn_top_bot(dir::cw, indicator, cube_sides::U, current_time);
		break;
		
	case 'u':
		executed = turn_top_bot(dir::ccw, indicator, cube_sides::U, current_time);
		break;
		
	case 'D':
		executed = turn_top_bot(dir::cw, indicator, cube_sides::D, current_time);
		break;
		
	case 'd':
		executed = turn_top_bot(dir::ccw, indicator, cube_sides::D, current_time);
		break;

	case 'X':
		if(indicator == 0)	//X0 represents to sliders moving, X1, X2,... represents the two grippers of said axis to move
			executed = slide(slider_X_, dir::open, current_time);
		else
		{
			executed = turn_side(gripper_L_, dir::ccw, indicator, current_time) & turn_side(gripper_R_, dir::cw, indicator, current_time);
		}
		break;

	case 'x':
		if(indicator == 0)
			executed = slide(slider_X_, dir::close, current_time);
		else
		{
			executed = turn_side(gripper_L_, dir::cw, indicator, current_time) & turn_side(gripper_R_, dir::ccw, indicator, current_time);
		}
		break;

	case 'Y':
		if (indicator == 0)
			executed = slide(slider_Y_, dir::open, current_time);
		else
		{
			executed = turn_side(gripper_F_, dir::cw, indicator, current_time) & turn_side(gripper_B_, dir::ccw, indicator, current_time);
		}
		break;

	case 'y':
		if (indicator == 0)
			executed = slide(slider_Y_, dir::close, current_time);
		else
		{
			executed = turn_side(gripper_F_, dir::ccw, indicator, current_time) & turn_side(gripper_B_, dir::cw, indicator, current_time);
		}
		break;
		break;

	case 'E':
		if(indicator == 1)		//E1
			executed = enable_steppers();
		else if(indicator == 2)		//E2
			executed = disable_steppers();
		else
		{
			Serial.write('NUL');
			return true;
		}
		break;

	case 'A':
		if(!adjust_cmd(indicator, dir::cw))
		{
			Serial.write('NUL');
			return true;
		}
		executed = true;
		break;

	case 'a':
		if(!adjust_cmd(indicator, dir::ccw))
		{
			Serial.write('NUL');
			return true;	//in order to avoid setting executed to true, but still signaling the end of the reading process
		}
		executed = true;	//correct command read
		break;

	//only testing
	
	default:
		Serial.write('NUL');
		return true; //finished executing command, even ignoring a wrong command is considered executing it
	}

	if(executed)	
	{
		Serial.write('ACK');	//sending ACK 
	}

	return executed;	//returns true if done moving
	
}





bool Solver::enable_steppers()
{
	gripper_L_.enable(true);
	slider_Y_.enable(true);
	gripper_R_.enable(true);
	gripper_F_.enable(true);
	gripper_B_.enable(true);
	slider_X_.enable(true);
	return true;
}

bool Solver::disable_steppers()
{
	gripper_L_.enable(false);
	slider_Y_.enable(false);
	gripper_R_.enable(false);
	gripper_F_.enable(false);
	gripper_B_.enable(false);
	slider_X_.enable(false);
	return true;
}

//--------------------------------------- actual moving functions -----------------------------------------------------------------------

bool Solver::turn_side(Gripper & gripper, dir dir, int turns, unsigned long time)
{
	if(!gripper.moving)	//if this is the first call of the function
	{
		gripper.set_turns(turns, dir);
		gripper.moving = true;
		return false;
	}
	else
	{
		if(gripper.update(time))
		{
			gripper.moving = false;
			return true;	//done moving
		}
		else
		{
			return false;	//still moving
		}
			
	}
}

bool Solver::slide(Slider & slider, dir dir, unsigned long time)
{
	static bool busy = false;

	if(!busy)	//if this is the first call of the function
	{
		slider.set_direction(dir);
		busy = true;
		return false;
	}
	else
	{
		if(slider.update(time))
		{
			busy = false;
			return true;	//done moving
		}
		else
		{
			return false;	//still moving
		}
	}
}



bool Solver::turn_top_bot(dir direction, int turns, cube_sides side, unsigned long time)
{
	
	switch(moving_state_top_bot_)
	{
		case state::unlock1:
			if(slide(slider_X_, dir::open, time))	//moving steppers away
				moving_state_top_bot_ = state::flip1;
			break;
			
		case state::flip1:
			if(turn_side(gripper_F_,dir::ccw,1, time) & turn_side(gripper_B_,dir::cw,1, time))
				moving_state_top_bot_ = state::lock1;
			break;

		case state::lock1:
			if(slide(slider_X_, dir::close, time))
				moving_state_top_bot_ = state::unlock2;
			break;

		case state::unlock2:
			if(slide(slider_Y_, dir::open, time))
				moving_state_top_bot_ = state::flip2;
			break;

		case state::flip2:
			if(turn_side(gripper_F_,dir::ccw,1, time) & turn_side(gripper_B_,dir::cw,1, time))
				moving_state_top_bot_ = state::lock2;
			break;

		case state::lock2:
			if(slide(slider_Y_, dir::close, time))
				moving_state_top_bot_ = state::turn;
			break;

		case state::turn:
			if(side == cube_sides::U)
			{
				if(turn_side(gripper_L_,direction,turns, time))
					moving_state_top_bot_ = state::unlock3;
			}
			if(side == cube_sides::D)
			{
				if(turn_side(gripper_R_,direction,turns, time))
					moving_state_top_bot_ = state::unlock3;
			}
			break;

		case state::unlock3:
			if(slide(slider_X_, dir::open, time))
					moving_state_top_bot_ = state::flip3;
			break;

		case state::flip3:
			if(turn_side(gripper_F_,dir::cw,1, time) & turn_side(gripper_B_,dir::ccw,1, time))
				moving_state_top_bot_ = state::turn_back;
			break;

		case state::turn_back:
		{
			dir n_dir;					//reversed direction
			if(direction == dir::cw)	//switching direcition to turn gripper back into initial position
					n_dir = dir::ccw;
				else
					n_dir = dir::cw;

			if(side == cube_sides::U)
			{
				if(turn_side(gripper_L_,n_dir,turns % 2, time))
					moving_state_top_bot_ = state::lock3;
			}
			if(side == cube_sides::D)
			{
				if(turn_side(gripper_R_,n_dir,turns % 2, time))
					moving_state_top_bot_ = state::lock3;
			}
		}
		break;

		case state::lock3:
			if(slide(slider_X_, dir::close, time))
				moving_state_top_bot_ = state::unlock4;
			break;

		case state::unlock4:
			if(slide(slider_Y_, dir::open, time))
				moving_state_top_bot_ = state::flip4;
			break;

		case state::flip4:
			if(turn_side(gripper_F_,dir::cw,1, time) & turn_side(gripper_B_,dir::ccw,1, time))
				moving_state_top_bot_ = state::lock4;
			break;

		case state::lock4:
			if(slide(slider_Y_, dir::close, time))
			{
				moving_state_top_bot_ = state::unlock1;
				return true;	//done moving
			}
			break;

	}
	return false;
}

bool Solver::adjust_cmd(int indicator, dir dir)
{
	switch(indicator)
	{
		case 1:
			slider_X_.adjust(dir);
			return true;

		case 2:
			slider_Y_.adjust(dir);
			return true;

		case 3:
			gripper_L_.adjust(dir);
			return true;

		case 4:
			gripper_R_.adjust(dir);
			return true;

		case 5:
			gripper_F_.adjust(dir);
			return true;

		case 6:
			gripper_B_.adjust(dir);
			return true;

		default:
			return false; //unknown command
	break;
	}
}



bool Solver::turn_back(Gripper &gripper, Slider &slider, dir direction, int turns, unsigned long time) 
{
	dir n_dir;						//reversed direction
	if(direction == dir::cw)	//switching direcition to turn gripper back into initial position
			n_dir = dir::ccw;
		else
			n_dir = dir::cw;

	switch(moving_state_turn_back_)
	{
		case state::turn:
			if(turn_side(gripper,direction, turns, time))
				moving_state_turn_back_ = state::unlock1;
			break;

		case state::unlock1:
			if(slide(slider, dir::open, time))
				moving_state_turn_back_ = state::turn_back;
			break;

		case state::turn_back:
			if(turn_side(gripper, n_dir, turns % 2, time))
				moving_state_turn_back_ = state::lock1;
			break;

		case state::lock1:
			if(slide(slider, dir::close, time))
			{
				moving_state_turn_back_ = state::turn;
				return true;	//done moving
			}
			break;

	}
	return false;
}






