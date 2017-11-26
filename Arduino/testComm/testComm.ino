#include <bacarComm.h>

// Définit l'objet associé au canal de communication avec l'Orange PI
BacarComm comm;
// Contiendra l'état actuel de la LED
bool ledState;


void setup() {
  // Configure la LED pour pouvoir l'utiliser dans loop()
  pinMode(LED_BUILTIN, OUTPUT);
  ledState = LOW;
  // Initialise l'objet comm
  comm.begin();
}
  int32_t x = 0 , y = 0 ;
  float u = 0, v = 0 ;

void loop() {

  
  // Vérifie si un nouveau message de l'Orange PI a été reçu
  if (comm.newMessage() == true) {
    // On lit les 4 valeurs contenues dans le message
    x = comm.xRead();
    y = comm.yRead();
    u = comm.uRead();
    v = comm.vRead();
    // et on les renvoie à l'Orange PI
    comm.sendMessage(x, y, u, v);
    // On change l'état de la LED pour indiquer qu'on a bien reçu le message
    ledState = not(ledState);
    digitalWrite(LED_BUILTIN, ledState);
  }


  
  
}
