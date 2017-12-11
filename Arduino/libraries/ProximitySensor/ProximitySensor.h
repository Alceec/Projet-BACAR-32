#ifndef PROXIMITYSENSOR_H
#define PROXIMITYSENSOR_H

#include <Arduino.h> 

#define MAX_VALUE 1023
#define CONSTANT 1

class ProximitySensor {
    public : 
        ProximitySensor(int Pin_Value) ; 
        bool Is_There_Object() ; 
    private : 
        int _pin_value ; 
};

#endif