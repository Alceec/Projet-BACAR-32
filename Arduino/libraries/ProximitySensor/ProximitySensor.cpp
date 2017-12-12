#include<ProximitySensor.h>
#include<Arduino.h> 

ProximitySensor::ProximitySensor(int Pin_Value, int Enable_Pin) {
    /* Pin Value is the output from the sensor, Enable_Pin sends 
    to the sensor the message to turn on or off*/
    _pin_value = Pin_Value;
    pinMode(_pin_value, INPUT);
    delayMicroseconds(210) ; 
}

bool ProximitySensor::Is_There_Object()
{ 
    if (!digitalRead(_pin_value)) 
    {
        delayMicroseconds(395) ; 
        return (digitalRead(_pin_value))? false : true;  
    }
    else 
        return false ; 
}
