#include <Controler.h>
#include <bacarComm.h>
#include <bacarMotor.h>
#include <ProximitySensor.h> 

ProximitySensor ProxiSens = ProximitySensor(20, 1) ;
BacarComm comm;
BacarMotor mot1 (10, 11, 12); 
BacarMotor mot2 (9, 7, 8)
Controler control = Controler(mot1, mot2s); 

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
  
  if (ProxiSens.Is_There_Object() ) {
    comm.sendMessage(0, 1, 0, 0);
    control.Move(0, 0); 
  }

}
