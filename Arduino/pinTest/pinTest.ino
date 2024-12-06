//
//Mikhail Kandel
// SLIM synchronization
// (not the pin test)
int armPin = 4;  
int triggerPin = 2;        
int trigGND = 5;          
int bnsPin = 5;
int led1 = 11;
int led2 = 10;
int led3 = 9;
int threshold = 300;
void leds(boolean one,boolean two, boolean three)
{
  digitalWrite(led1,one);
  digitalWrite(led2,two);
  digitalWrite(led3,three);
}
boolean TLOW = LOW;
boolean THIGH = HIGH;
void setup()
{
  Serial.begin(9600);
  pinMode(triggerPin, OUTPUT);
  digitalWrite(triggerPin,TLOW);
  pinMode(trigGND, OUTPUT);
  digitalWrite(trigGND,LOW);
  pinMode(armPin, INPUT);
  pinMode(bnsPin, INPUT);
  pinMode(led1,OUTPUT);
  pinMode(led2,OUTPUT);
  pinMode(led3,OUTPUT);
}
int waitforPin(int pin)
{
  int v=0;
  while((v=analogRead(pin)) < threshold)
  {
    int nop=1;
  }
  return v;
}
void setExposure(byte time)
{
  digitalWrite(triggerPin,THIGH);
  delay(time);
  digitalWrite(triggerPin,TLOW);
}
void pulseExposure()
{
    digitalWrite(triggerPin,THIGH);
    delay(1);
    digitalWrite(triggerPin,TLOW);
}
byte getmessage()
{
  while (Serial.available() <1)
  {

  }
  return Serial.read();
}
void waitMessage(byte m)
{
  while (getmessage()!=m)
  {
  }
}
void sendMessage(byte v)
{
  Serial.write( v);
}

void loop()
{
  byte slm = getmessage();
  byte expo = getmessage();
  delay(slm);
  waitforPin(armPin);
//  pulseExposure();
  setExposure(expo);
}

