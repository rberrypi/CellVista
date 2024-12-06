//
//Mikhail Kandel
//Waits for a serial signal to begin firing the arduino device
int led1Pin = 8;
int led2Pin = 9;
int led3Pin = 10;
int triggerPin = 11;
int armPin = 0;
//
//Setup
//
void offLeds()
{
  digitalWrite(led1Pin,LOW);
  digitalWrite(led2Pin,LOW);
  digitalWrite(led3Pin,LOW);
}
void setup() 
{
  Serial.begin(9600);
  pinMode(led1Pin,OUTPUT);
  pinMode(led2Pin,OUTPUT);
  pinMode(led3Pin,OUTPUT);
  pinMode(triggerPin,OUTPUT);
  offLeds();
  digitalWrite(triggerPin,LOW);
}
void onPattern()
{
  digitalWrite(led1Pin,HIGH);
  digitalWrite(led2Pin,HIGH);
  digitalWrite(led3Pin,HIGH);
}
void waitForComputer()
{
  //ping
  Serial.write("ST");//race condition with the comptuer
  while (0==Serial.available())
  {
  }
  //Pong
  int bitBucket = Serial.read();
}
//
//Triggering
//
void expose(byte exposure)
{
  /*
  In overlap mode, every positive edge of an external trigger will trigger a frame read out and start a new exposure for the
   next frame. The period of external trigger pulse defines exposure and cycle time for each frame read out. ~ 37/67 hw guide
   */
  digitalWrite(triggerPin,HIGH);
  delay(exposure);
  digitalWrite(triggerPin,LOW);
}
void waitForArm()
{
  int sensorValue = analogRead(A0);
  while (true)
  {
    if (sensorValue> 500)
    {
      digitalWrite(triggerPin,HIGH);
      delay(1);
      digitalWrite(triggerPin,LOW);
      delay(1);//lazy man's debounce
      return;
    }
    sensorValue = analogRead(A0);
  }
}
void waitForSignal(byte Signal)
{//called 'bad programming'
  byte byteRead=0;
  while( byteRead != Signal)
  {
    if (Serial.available()) 
    {
      byteRead = Serial.read();
    }
  }
}
byte waitForSignal()
{
  while (Serial.available() == 0)
  {
    int nop=0;
  }
  return incomingByte = Serial.read();
}
void notifyDone()
{
  Serial.write("D");
}

void loop() 
{
  //onPattern();
  //Serial.write("ST");//race condition with the comptuer
  ///*
  waitForComputer();
  digitalWrite(led3Pin,LOW);
  waitForSignal('S');
  offLeds();
  while (true)
  {
    //
    digitalWrite(led1Pin,HIGH);
    int exposure = waitForSignal();
    digitalWrite(led1Pin,LOW);
    //
    digitalWrite(led2Pin,HIGH);
    waitForArm();
    digitalWrite(led2Pin,LOW);
    //
    digitalWrite(led3Pin,HIGH);
    expose(exposure);
    notifyDone();
    digitalWrite(led3Pin,LOW);
  }
  //*/
}


