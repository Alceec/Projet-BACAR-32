#include <Controler.h>
#include <bacarComm.h>
#include <bacarMotor.h>
#include <ProximitySensor.h> 

ProximitySensor ProxiSens = ProximitySensor(20) ;
BacarComm comm;
Controler control = Controler(BacarMotor(10, 11, 12), BacarMotor(9, 7, 8)); 

void setup() {
  comm.begin();
}


void loop() {
  //getting the info
  // x is the radius distance in cm
  // v is the angular speed 
  if ( comm.newMessage() == true ) {
    control.Move (comm.vRead(), float(comm.xRead()) * 1000);
  }
  
  comm.sendMessage(0, ProxiSens.Is_There_Object(), 0, 0) ;

}
