const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int EnX = 8;
//
//int val_step = 0;
//int val_dir = 0;
//
//void setup() {
//  pinMode(EnX, OUTPUT);
//  pinMode(StepX, OUTPUT);
//  pinMode(DirX, OUTPUT);
//  pinMode(StepY,INPUT);
//  pinMode(DirY,INPUT);
//  digitalWrite(EnX, LOW);
//  digitalWrite(StepX, LOW);
//  digitalWrite(DirX, LOW);
//  digitalWrite(StepY,LOW); 
//  digitalWrite(DirY,LOW); 
//}
//
//void loop() {  
//  if(val_step != digitalRead(StepY)){
//    val_step = digitalRead(StepY);  
//    digitalWrite(StepX, val_step); 
//  }
//
//  if(val_dir != digitalRead(DirY)){
//    val_dir = digitalRead(DirY);  
//    digitalWrite(DirX, val_dir); 
//  }
//}

void setup() {
  pinMode(EnX, OUTPUT);
  pinMode(StepX, OUTPUT);
  pinMode(DirX, OUTPUT);

  digitalWrite(EnX, LOW);
  digitalWrite(StepX, LOW);
  digitalWrite(DirX, LOW);
}

void loop() {
  digitalWrite(StepX, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(20);                       // wait for a second
  digitalWrite(StepX, LOW);    // turn the LED off by making the voltage LOW
  delay(20);                       // wait for a second
}
