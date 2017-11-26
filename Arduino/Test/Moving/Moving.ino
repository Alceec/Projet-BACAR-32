#include <Controler.h>
#include <bacarComm.h>
#include <bacarMotor.h>



BacarComm comm;
BacarMotor motor1(9, 7, 8);
BacarMotor motor2(10, 11, 12);


void setup() {
  comm.begin();
  motor1.begin();
  motor2.begin();
  Main main;
  main.Print();

}

int32_t Mode = 0, y = 0 ;
float Speed = 0,  Angular_speed = 0 ;

//the ratio between the value of u and
//the value actuate() takes
float motorSpeedRatio = 1;
//
void loop() {


  //getting the info
  if ( comm.newMessage() == true ) {
    Mode = comm.xRead();
    y = comm.yRead();
    Speed = comm.uRead();         //unit : m/s
    Angular_speed = comm.vRead(); //unit : deg/s
  }
  //Mode 0 is just a simple rotation
  if ( Mode == 0 ) {

  }
  else if ( Mode == 1 ) {

  }


}
