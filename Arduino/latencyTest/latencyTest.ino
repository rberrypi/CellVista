//
//Mikhail Kandel
void setup()
{
  Serial.begin(9600);
}
byte getMessage()
{
  while (Serial.available() <1)
  {

  }
  return Serial.read();
}
void waitMessage(byte m)
{
  while (getMessage()!=m)
  {
  }
}
void sendMessage(byte v)
{
  Serial.write( v);
}

void loop()
{
  waitMessage(80);
  sendMessage(80);
}

