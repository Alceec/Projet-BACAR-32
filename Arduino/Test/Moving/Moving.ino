#include <Controler.h>
#include <bacarComm.h>
#include <bacarMotor.h>



BacarComm comm;
Controler control = Controler(BacarMotor motor1(9, 7, 8), BacarMotor motor2(10, 11, 12)); 

void setup() {
  comm.begin();
  motor1.begin();
  motor2.begin();
  main.Print();

}


void loop() {
  //getting the info
  // x is the radius distance in cm
  // v is the angular speed 
  if ( comm.newMessage() == true ) {
    control.Move (comm.vRead(), float(comm.xRead()) / 1000);
  }


}
