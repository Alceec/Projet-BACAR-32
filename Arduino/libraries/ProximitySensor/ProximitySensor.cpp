#include<ProximitySensor.h>

ProximitySensor::ProximitySensor(int Pin_Value) {
    _pin_value = Pin_Value;
}

float ProximitySensor::Read_Distance()
{
    int val = analogRead(_pin_value); 
    return sqrt((float) (CONSTANT / (MAX_VALUE - val))) ;
}
