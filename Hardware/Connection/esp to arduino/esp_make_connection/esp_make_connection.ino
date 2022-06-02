//
//  ESP32 code
//

#define RXp2 16
#define TXp2 17

void setup()
{
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial2.begin(115200, SERIAL_8N1, RXp2, TXp2);
}
void loop()
{

  while (Serial.available() == 0)
  { // Wait for user input
  }
  String command = Serial.readString(); // Reading the Input string from Serial port.
  Serial2.println(command);

  delay(2000);
}
