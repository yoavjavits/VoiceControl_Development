#include <Stepper.h>

const int X_ENABLE_PIN       = 38;
const int Y_ENABLE_PIN       = A2;

const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution

Stepper myStepper1(stepsPerRevolution, A0, A1);
Stepper myStepper2(stepsPerRevolution, A6, A7);

void setup() {
  // set the speed at 60 rpm:
  myStepper1.setSpeed(120);
  myStepper2.setSpeed(120);

  pinMode(X_ENABLE_PIN, OUTPUT);
  pinMode(Y_ENABLE_PIN, OUTPUT);

  digitalWrite(X_ENABLE_PIN, LOW);
  digitalWrite(Y_ENABLE_PIN, LOW);
  pinMode(LED_BUILTIN, OUTPUT);

  // initialize the serial port:
  Serial.begin(9600);
  delay(2000);
}

void loop() {
  // step one revolution  in one direction:
  Serial.println("clockwise");
  myStepper1.step(stepsPerRevolution);
  myStepper2.step(stepsPerRevolution);
  digitalWrite(LED_BUILTIN, LOW);
  delay(2000);

  // step one revolution in the other direction:
  Serial.println("counterclockwise");
  myStepper1.step(-stepsPerRevolution);
  myStepper2.step(-stepsPerRevolution);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(2000);
}
