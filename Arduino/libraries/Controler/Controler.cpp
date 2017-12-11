#include <bacarComm.h>
#include "Controler.h"

using namespace std;

Controler::Controler(BacarMotor mot1 , BacarMotor mot2 ) {
    motor1 = &mot1 ; // LEFT MOTOR
    motor2 = &mot2 ; // RIGHT MOTOR
}

void Controler::Move( float Angular_Speed, float radius_from_center ) {  // positive to turn right
    /* Angular_Speed (rad/s) , radius_from_center (m) */
    if ( radius_from_center != INF ) {

        float radius1 = (_CAR_WIDTH / 2 ) + radius_from_center ; 

        float speed_mot1 = (Angular_Speed * radius1) / _SPEED_AT_FULL_CAPACITY ; 

        float radius2 = (_CAR_WIDTH / 2 ) - radius_from_center ; 

        float speed_mot2 = (Angular_Speed * radius2) / _SPEED_AT_FULL_CAPACITY ; 

        (*motor1).actuate( speed_mot1 ) ; 
        (*motor2).actuate( speed_mot2 ) ; 
    }
    else {
        float speed = Angular_Speed / _SPEED_AT_FULL_CAPACITY;
        (*motor1).actuate(speed / 2.0) ;
        (*motor2).actuate(speed / 2.0) ;
    }
    



}
