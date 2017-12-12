#ifndef PROXIMITYSENSOR_H
#define PROXIMITYSENSOR_H

#include <Arduino.h> 

#define MAX_VALUE 1023
#define CONSTANT 1

class ProximitySensor {
    public : 
        ProximitySensor(int Pin_Value, int Enable_Pin) ; 
        bool Is_There_Object() ; 
    private : 
        int _pin_value ; 
		int _enable_pin; 
};

#endif