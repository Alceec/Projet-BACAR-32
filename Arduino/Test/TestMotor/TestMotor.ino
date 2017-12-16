#include <bacarMotor.h>

const int SensorDistance = 6;

BacarMotor mot1 (10, 11, 12); 
BacarMotor mot2 (9, 7, 8);


void setup () {
    mot1.begin();
    mot2.begin();
    pinMode(SensorDistance, INPUT); 
    Serial.begin(9600);
}

void loop() {
    mot1.actuate(0.5) ; 
    mot2.actuate(0.5) ; 
    if (!digitalRead(SensorDistance))
        {
            mot1.halt(); 
            mot2.halt();
            delay(5000); 
        }
}
