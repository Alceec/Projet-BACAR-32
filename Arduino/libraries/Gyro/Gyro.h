#ifndef GYRO_H
#define GYRO_H

class Gyro {

    private :
        float _current_rotation = 0. ;
        float _cap = 0. ;               //the angle it is supposed to have
        float _precision = 0.5;

        //=========================
        unsigned int _gyroPin;
        float _gyroVoltage, _gyroZeroVoltage, _gyroSensitivity, _rotationThreshold ;
        //=========================
        void Update_Current_Rotation();

    public :
        Gyro(unsigned int gyroPin, float gyroVoltage, float gyroZeroVoltage, float gyroSensitivity, float rotationThreshold ) ;
        void New_Rotation(float angle_to_do ) ;
        bool Has_rotated();
        void Reinitialize_cap();

}

#endif
