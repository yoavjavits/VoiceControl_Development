#ifndef HARDWARE_PARAMETERS_H
#define HARDWARE_PARAMETERS_H

//------------------ general ------------------
static constexpr int steps_per_rot = 200;
static constexpr int gripper_speed = 190;
static constexpr int slider_speed = 270;
static constexpr int max_angle = 250;

static constexpr long baudrate = 115200;

//----------------- gripper F ----------------- MOT4 external motor
static constexpr int gripperF_step = 32;
static constexpr int gripperF_dir = 47;
static constexpr int gripperF_en = 45;

//----------------- gripper B ----------------- MOT1 E1 motor
static constexpr int gripperB_step = 36;
static constexpr int gripperB_dir = 34;
static constexpr int gripperB_en = 30;

//----------------- gripper L ----------------- MOT6 E0 motor
static constexpr int gripperL_step = 26;
static constexpr int gripperL_dir = 28;
static constexpr int gripperL_en = 24;

//----------------- gripper R ----------------- MOT3 Z motor
static constexpr int gripperR_step = 46;
static constexpr int gripperR_dir = 48;
static constexpr int gripperR_en = A8;

//----------------- slider y ----------------- MOT5 Y motor
static constexpr int sliderY_step = A6;
static constexpr int sliderY_dir = A7;
static constexpr int sliderY_en = A2;

//----------------- slider x ----------------- MOT2 X motor
static constexpr int sliderX_step = A0;
static constexpr int sliderX_dir = A1;
static constexpr int sliderX_en = 38;

#endif
