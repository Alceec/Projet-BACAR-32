#include <bacarComm.h>

#define LED 13

BacarComm comm;

void setup() {
  // put your setup code here, to run once:
  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);
  comm.begin();
}


void loop() {
  int32_t x, y;
  float u, v;
  
  // put your main code here, to run repeatedly:
  if (comm.newMessage() == true) {
    digitalWrite(LED, HIGH);
    x = comm.xRead();
    y = comm.yRead();
    u = comm.uRead();
    v = comm.vRead();
    comm.sendMessage(0, x, 0, u);
    comm.sendMessage(1, y, 1, v);
    comm.sendMessage(-x, -1, -u, -1);  
  }
}
