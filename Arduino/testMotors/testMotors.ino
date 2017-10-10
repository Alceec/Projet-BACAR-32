#include <bacarMotor.h>

/* Définit les objets liés aux moteurs de la voiture
 * Les paramètres définissent les pattes auxquelles sont 
 * raccordées le H-bridge.
 * Le 1er paramètres définit la patte PWM qui doit être
 * la patte D9 ou D10.
 * Les deux autres paramètres définissent les pattes IN1 et IN2,
 * ce peut être n'importe quelle patte digitale de l'Arduino */
BacarMotor motorA(9, 7, 8);
BacarMotor motorB(10, 11, 12);


void setup() {
  // Configure la LED pour pouvoir l'utiliser dans loop()
  pinMode(LED_BUILTIN, OUTPUT);
  // Initialise les objets moteurs
  motorA.begin();
  motorB.begin();
}


void loop() {
  /* On commence par activer les moteurs en appliquant la moitié
   * de la tension d'alimentation du H-bridge.
   * On allume la LED pour indiquer que les moteurs devraient 
   * être en train de tourner. */
  digitalWrite(LED_BUILTIN, HIGH);
  motorA.actuate(0.5);
  motorB.actuate(0.5);
  // On attend 1sec avant d'arrêter les moteurs
  delay(1000);
  /* On peut arrêter un moteur de 2 manières :
   *   en utilisant halt()
   *   en lui appliquant une tension nulle. */
  motorA.halt();
  motorB.actuate(0);
  // On éteint la LED pour indiquer l'arrêt des moteurs
  digitalWrite(LED_BUILTIN, LOW);
  // On attend 5sec avant que loop() ne recommence
  delay(5000);
}

