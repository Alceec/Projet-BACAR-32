#include "Gyro.h"
#include <Arduino.h>
#include <cmath>

#define TIME_BETWEEN_LOOP 0.01

using namespace std ;

Gyro::Gyro(unsigned int gyroPin, float gyroVoltage, float gyroZeroVoltage, float gyroSensitivity, float rotationThreshold ) {
  _gyroPin = gyroPin;
  _gyroVoltage = gyroVoltage;
  _gyroZeroVoltage = gyroZeroVoltage ;
  _gyroSensitivity = gyroSensitivity;   //how many Volts equal 1 (deg/s)
  _rotationThreshold = rotationThreshold;

}

void Gyro::Update_Current_Rotation(){
  //convert the signal in bits to voltage
  float gyroRate = (analogRead(_gyroPin) * _gyroVoltage ) / 1023;
  //scale to the beginning position
  gyroRate -= _gyroZeroVoltage;
  //see the angular speed
  gyroRate  /= _gyroSensitivity ;  //unit : deg/s

  if(abs(gyroRate) >= (rotationThreshold)){
    gyroRate *= TIME_BETWEEN_LOOP;
    _current_rotation += gyroRate;
  }

}

void Gyro::New_Rotation(float angle_to_do) {
  _cap = _current_rotation + angle_to_do;
}

void Gyro::Reinitialize_cap(){
  _cap = _current_rotation;
}


bool Gyro::Is_(){
  Update_Current_Rotation()
  return (_current_rotation)
}
