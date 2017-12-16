#include <Controler.h>
#include <bacarComm.h>
#include <bacarMotor.h>

const int SensorPin = 6;

BacarComm comm;
BacarMotor mot1 (10, 11, 12); 
BacarMotor mot2 (9, 7, 8);
Controler control(mot1, mot2); 

void setup() {
  comm.begin();
  Serial.begin(9600); 

}


void loop() {
  
  //getting the info
  // x is the radius distance in cm
  // v is the angular speed in rad/s
  if ( comm.newMessage() == true ) {
    //control.Move (comm.vRead(), float(comm.xRead()) / 1000);
    int radius = comm.xRead() / 1000; 
    float _speed = comm.vRead(); 
    control.Move(_speed, radius) ; 
  }

  
  
  if (!digitalRead(SensorPin)) { 
    Serial.print("Seeing a wall\n"); 
    comm.sendMessage(0, 1, 0, 0);
    control.Move(0, 0);
    delay(5000); 
  }

}



