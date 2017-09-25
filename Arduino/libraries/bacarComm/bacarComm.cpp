#include "bacarComm.h"

//********************************************************
// Low level functions prototypes
//********************************************************
int32_t getSerialBinaryInt();
float getSerialBinaryFloat();


//********************************************************
// Class implementation
//********************************************************
BacarComm::BacarComm() {
}

void BacarComm::begin(void) {
  //----- Set up serial port
  Serial.begin(115200);
  while(Serial.available()) {
    Serial.read();
  }
  _u = 0;
  _v = 0;
  _x = 0;
  _y = 0;
}

int32_t BacarComm::xRead(void) {
  return(_x);
}

int32_t BacarComm::yRead(void) {
  return(_y);
}

float BacarComm::uRead(void) {
  return(_u);
}

float BacarComm::vRead(void) {
  return(_v);
}

bool BacarComm::newMessage(void) {
  if (Serial.available() >= 16) {
    _x = getSerialBinaryInt();
    _y = getSerialBinaryInt();
    _u = getSerialBinaryFloat();
    _v = getSerialBinaryFloat();
    return (true);
  } else {
    return(false);
  }
}

void BacarComm::sendMessage(int32_t x, int32_t y, float u, float v) {
  const char* HEAD="EHLO";
  int HEADLENGTH = 4;
  Serial.write((byte*)(HEAD), HEADLENGTH);
  Serial.write((byte*)&x, sizeof(int32_t));
  Serial.write((byte*)&y, sizeof(int32_t));
  Serial.write((byte*)&u, sizeof(float));
  Serial.write((byte*)&v, sizeof(float));
  Serial.flush();
}

//********************************************************
// Low level functions implementation
//********************************************************

int32_t getSerialBinaryInt() {
  static byte buff[sizeof(int32_t)];
  static int result;
  int tmp;
  if (Serial.available() > 0)
    for (unsigned int i=0; i < sizeof(int32_t); i++) {
      tmp = Serial.read();
      buff[i] = (byte) tmp;
    }
  //On some embedded chips cassting buff as an int will cause exceptions
  memcpy(&result, buff, sizeof(int));
  return result;
}

float getSerialBinaryFloat() {
  static byte buff[sizeof(float)];
  static float result;
  int tmp;
  if (Serial.available() > 0) {
    for (unsigned int i=0; i < sizeof(float); i++) {
      tmp = Serial.read();
      buff[i] = (byte) tmp;
    }
  }
  //On some embedded chips casting buff as an int will cause exceptions
  memcpy(&result, buff, sizeof(float));
  return result;
}
