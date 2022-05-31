#include <Stepper.h>

const int X_ENABLE_PIN       = 38;
const int Y_ENABLE_PIN       = A2;
const int Z_ENABLE_PIN       = A8;
const int E0_ENABLE_PIN       = 24;
const int E1_ENABLE_PIN       = 30;


const int O_STEP_PIN         = 45;
const int O_DIR_PIN          = 47;
const int O_ENABLE_PIN       = 32;


const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution

Stepper myStepperX(stepsPerRevolution, A0, A1); // X axis stepper
Stepper myStepperY(stepsPerRevolution, A6, A7); // Y axis stepper
Stepper myStepperZ(stepsPerRevolution, 46, 48); // Z axis stepper
Stepper myStepperE0(stepsPerRevolution, 26, 28); // E0 axis stepper
Stepper myStepperE1(stepsPerRevolution, 36, 34); // E1 axis stepper
Stepper myStepperO(stepsPerRevolution, O_STEP_PIN, O_DIR_PIN); // O axis stepper


void setup() {
  // set the speed at 60 rpm:
  myStepperX.setSpeed(120);
  myStepperY.setSpeed(120);
  myStepperZ.setSpeed(120);
  myStepperE0.setSpeed(120);
  myStepperE1.setSpeed(120);
  myStepperO.setSpeed(120);

  pinMode(X_ENABLE_PIN, OUTPUT);
  pinMode(Y_ENABLE_PIN, OUTPUT);
  pinMode(Z_ENABLE_PIN, OUTPUT);
  pinMode(E0_ENABLE_PIN, OUTPUT);
  pinMode(E1_ENABLE_PIN, OUTPUT);
  pinMode(O_ENABLE_PIN, OUTPUT);

  digitalWrite(X_ENABLE_PIN, LOW); 
  digitalWrite(Y_ENABLE_PIN, LOW);
  digitalWrite(Z_ENABLE_PIN, LOW);
  digitalWrite(E0_ENABLE_PIN, LOW);
  digitalWrite(E1_ENABLE_PIN, LOW);
  digitalWrite(O_ENABLE_PIN, LOW);
  pinMode(LED_BUILTIN, OUTPUT);
  
  // initialize the serial port: 
  Serial.begin(9600);
  delay(2000);
}

void loop() {
  // step one revolution  in one direction:
  Serial.println("clockwise");
//  myStepperX.step(stepsPerRevolution);
//  myStepperY.step(stepsPerRevolution);
//  myStepperZ.step(stepsPerRevolution);
//  myStepperE0.step(stepsPerRevolution);
//  myStepperE1.step(stepsPerRevolution);
  myStepperO.step(stepsPerRevolution);
  digitalWrite(LED_BUILTIN, LOW);
  delay(2000);

  // step one revolution in the other direction:
  Serial.println("counterclockwise");
//  myStepperX.step(-stepsPerRevolution);
//  myStepperY.step(-stepsPerRevolution);
//  myStepperZ.step(-stepsPerRevolution);
//  myStepperE0.step(-stepsPerRevolution);
//  myStepperE1.step(-stepsPerRevolution);
  myStepperO.step(-stepsPerRevolution);
  digitalWrite(LED_BUILTIN, HIGH);
  
  delay(2000);
}
