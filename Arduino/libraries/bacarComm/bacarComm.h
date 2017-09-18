#ifndef BACARCOMM_H
#define BACARCOMM_H

#include <Arduino.h>


class BacarComm {
private:
  int32_t _x, _y;
  float _u, _v;

public:
  BacarComm ();

  void begin(void);

  bool newMessage(void);

  int32_t xRead(void);

  int32_t yRead(void);
    
  float uRead(void);

  float vRead(void);

  void sendMessage(int32_t x, int32_t y, float u, float v);
};

#endif /* end of include guard #ifndef BACARCOMM_H */
