//
//  arduino mega code
//


#define RXp2 16
#define TXp2 17

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial2.begin(115200);
}
void loop() {
   char data[100]={0};
   if (Serial2.available())
   {
    Serial2.readBytesUntil('\n',data,100);
    Serial.print("Received: ");
    Serial.println(data);
   }
}
