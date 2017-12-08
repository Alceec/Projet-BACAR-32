#ifndef PROXIMITYSENSOR_H
#define PROXIMITYSENSOR_H

#include <Arduino.h> 

#define MAX_VALUE 1028
#define CONSTANT 1

class ProximitySensor {
    public : 
        ProximitySensor(int Pin_Value) ; 
        float Read_Distance() ; 
    private : 
        int _pin_value ; 
};

#endif