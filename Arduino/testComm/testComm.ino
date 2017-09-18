#include <bacarComm.h>

BacarComm comm;

void setup() {
  // put your setup code here, to run once:
  comm.begin();
}


void loop() {
  int32_t x, y;
  float u, v;
  // put your main code here, to run repeatedly:
  if (comm.newMessage() == true) {
    x = comm.xRead();
    y = comm.yRead();
    u = comm.uRead();
    v = comm.vRead();
    comm.sendMessage(0, u, 0, x);
    comm.sendMessage(1, v, 1, y);
    comm.sendMessage(-u, -1, -x, -1);  
  }
}
