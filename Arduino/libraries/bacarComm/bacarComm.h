#ifndef BACARCOMM_H
#define BACARCOMM_H

#include <Arduino.h>


class BacarComm {
private:
  int32_t _x, _y;
  float _u, _v;

public:
  /** Crée un objet bacarComm */
  BacarComm ();

  /** Initialise la communication avec l'Orange Pi.
   *  Cette fonction doit être appelée dans le setup du croquis. */
  void begin(void);

  /** Vérifie si un nouveau message de l'Orange Pi a été reçu.
   *  Si oui, le message est lu et les quatres paramètres du message (x,y,u,v) sont sauvés
   *  dans des champs privés de l'objet.
   *  Retour :
   *    true si un nouveau message a été reçu, false sinon. */
  bool newMessage(void);

  /** Renvoie la valeur du paramètre x du dernier message reçu */
  int32_t xRead(void);

  /** Renvoie la valeur du paramètre y du dernier message reçu */
  int32_t yRead(void);
    
  /** Renvoie la valeur du paramètre u du dernier message reçu */
  float uRead(void);

  /** Renvoie la valeur du paramètre v du dernier message reçu */
  float vRead(void);

  /** Envoie un message à l'Orange Pi.
   *  Paramètres :
   *    x,y,u,v : paramètres à envoyer dans le message. */
  void sendMessage(int32_t x, int32_t y, float u, float v);
};

#endif /* end of include guard #ifndef BACARCOMM_H */
