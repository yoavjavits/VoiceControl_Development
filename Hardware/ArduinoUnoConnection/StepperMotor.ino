const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int EnX = 8;

int val_step = 0;
int val_dir = 0;

void setup() {
  delay(3000);
  pinMode(EnX, OUTPUT);
  pinMode(StepX, OUTPUT);
  pinMode(DirX, OUTPUT);
  pinMode(StepY,INPUT);
  pinMode(DirY,INPUT);
  digitalWrite(EnX, LOW);
  digitalWrite(StepX, LOW);
  digitalWrite(DirX, LOW);
  digitalWrite(StepY,LOW); 
  digitalWrite(DirY,LOW); 
  delay(3000);
}

void loop() {  
  if(val_step != digitalRead(StepY)){
    val_step = digitalRead(StepY);  
    digitalWrite(StepX, val_step); 
  }

  if(val_dir != digitalRead(DirY)){
    val_dir = digitalRead(DirY);  
    digitalWrite(DirX, val_dir); 
  }
}
