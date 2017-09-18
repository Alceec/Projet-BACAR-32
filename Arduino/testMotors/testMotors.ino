#include <bacarMotor.h>


#define LED 13

BacarMotor motorA(9, 7, 8);
BacarMotor motorB(10, 11, 12);

void setup() {
  // put your setup code here, to run once:
  pinMode(LED, OUTPUT);
  motorA.begin();
  motorB.begin();
}


void loop() {
  digitalWrite(LED, HIGH);
  motorA.actuate(0.3);
  motorB.actuate(0.2);
  delay(1000);
  motorA.halt();
  motorB.halt();
  digitalWrite(LED, LOW);
  while(1) {
  }
}
