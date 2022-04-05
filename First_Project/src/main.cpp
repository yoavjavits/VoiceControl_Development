/*
  Blink with Function sketch
  function-blink.ino
  Use for PlatformIO demo

  DroneBot Workshop 2021
  https://dronebotworkshop.com
*/

#include <Arduino.h>
// Define LED pin
#define LED_PIN 2

void blink_led(int LED, int delaytime)
{
  // Set output HIGH for specified time
  digitalWrite(LED, HIGH);
  delay(delaytime);

  // Set output LOW for specified time
  digitalWrite(LED, LOW);
  delay(delaytime);
}

void setup()
{
  // Initialize LED pin as an output.
  pinMode(LED_PIN, OUTPUT);
}

void loop()
{
  // Blink the LED at half-second interval six times
  for (int i = 0; i < 6; i++)
  {
    blink_led(LED_PIN, 500);
  }

  // Blink the LED at two-second interval three times
  for (int i = 0; i < 3; i++)
  {
    blink_led(LED_PIN, 2000);
  }
}
