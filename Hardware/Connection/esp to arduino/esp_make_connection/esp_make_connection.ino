//
//  ESP32 code
//


#define RXp2 16
#define TXp2 17

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial2.begin(115200, SERIAL_8N1, RXp2, TXp2);
}
void loop() {

  Serial2.println("B1");
  //Serial.println("X2");
  delay(2000);

  
  /*for (int i =0; i<301; i++)
  {
    Serial2.println(i);
    //Serial.println(Serial2.readString());
    Serial.print("sent packet ");
    delay(1000);
    
  }*/
}
