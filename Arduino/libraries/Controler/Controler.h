#ifndef CONTROLER_H
#define CONTROLER_H

#include <bacarMotor.h>

class Controler {
  private :
    const float _SPEED_AT_FULL_CAPACITY = 1 ; // m/s 
    const float _CAR_WIDTH = 0.2 ; // metre 
    BacarMotor motor1, motor2;
  public :
    Controler(BacarMotor mot1, BacarMotor mot2);
    void Move( float Angular_Speed, float radius_from_center ); 
    
};

#endif
