#ifndef HARDWARE_PARAMETERS_H
#define HARDWARE_PARAMETERS_H

#pragma once
//------------------ general ------------------
static constexpr int steps_per_rot = 200;
static constexpr int gripper_speed = 190;
static constexpr int slider_speed = 270; //150;
static constexpr int max_angle = 250;

static constexpr long baudrate = 115200;
/*
//----------------- gripper F ----------------- MOT4
static constexpr int gripperF_step = 8;
static constexpr int gripperF_dir = 9;
static constexpr int gripperF_en = 39;

//----------------- gripper B ----------------- MOT1
static constexpr int gripperB_step = 2;
static constexpr int gripperB_dir = 3;
static constexpr int gripperB_en = 33;

//----------------- gripper L ----------------- MOT6
static constexpr int gripperL_step = 12;
static constexpr int gripperL_dir = 13;
static constexpr int gripperL_en = 37;

//----------------- gripper R ----------------- MOT3
static constexpr int gripperR_step = 6;
static constexpr int gripperR_dir = 7;
static constexpr int gripperR_en = 35;

//----------------- slider y ----------------- MOT5
static constexpr int sliderY_step = 10;
static constexpr int sliderY_dir = 11;
static constexpr int sliderY_en = 31;

//----------------- slider x ----------------- MOT2
static constexpr int sliderX_step = 4;
static constexpr int sliderX_dir = 5;
static constexpr int sliderX_en = 41;
*/

//----------------- gripper F ----------------- MOT4
static constexpr int gripperF_step = 26;
static constexpr int gripperF_dir = 28;
static constexpr int gripperF_en = 24;

//----------------- gripper B ----------------- MOT1
static constexpr int gripperB_step = A0;
static constexpr int gripperB_dir = A1;
static constexpr int gripperB_en = 38;

//----------------- gripper L ----------------- MOT6
static constexpr int gripperL_step = 45;
static constexpr int gripperL_dir = 47;
static constexpr int gripperL_en = 32;

//----------------- gripper R ----------------- MOT3
static constexpr int gripperR_step = 46;
static constexpr int gripperR_dir = 48;
static constexpr int gripperR_en = A8;

//----------------- slider y ----------------- MOT5
static constexpr int sliderY_step = 36;
static constexpr int sliderY_dir = 34;
static constexpr int sliderY_en = 30;

//----------------- slider x ----------------- MOT2
static constexpr int sliderX_step = A6;
static constexpr int sliderX_dir = A7;
static constexpr int sliderX_en = A2;


#endif
