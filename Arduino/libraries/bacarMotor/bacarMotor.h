#ifndef BACARMOTOR_H
#define BACARMOTOR_H

#include <Arduino.h>

class BacarMotor {
private:
  const int pwm_pin;         // pin on which to apply the PWM signal
  const int in1_pin;         // pin that dictates direction
  const int in2_pin;         // pin that dictates direction

  const int PWM_MAX = 255;
  const int PWM_MIN = PWM_MAX / 5; //if actuated, actuate at least for 20 %

  float _pwm_value; //pwm actuation value (between 0 and PWM_MAX)

public:
  BacarMotor (int pwm_pin, int in1_pin, int in2_pin);

  void begin();

  /** stop BacarMotor **/
  void halt();

  float getPwmValue();

  void actuate(float pwm_value);
};

#endif /* end of include guard BACARBacarMotor_H */
