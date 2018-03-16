 #include <bacarComm.h>
 BacarComm comm;

//Proximity sensor left and right
int IRMD = 6;
int IRMG = 5;
int IRMF = 4;
int valD;
int valG;
int valF;
//MOTOR A
int AIM1 = 7;
int AIM2 = 8;
int PWMA = 9;
//MOTOR B
int BIN1 = 11;
int BIN2 = 12;
int PWMB = 10;
//standby
int STBY = 3;


void setup() {
//Initialisation des moteurs
pinMode(LED_BUILTIN, OUTPUT);
pinMode(AIM1,OUTPUT);
pinMode(AIM2,OUTPUT);
pinMode(PWMA,OUTPUT);
pinMode(BIN1,OUTPUT);
pinMode(BIN2,OUTPUT);
pinMode(PWMB,OUTPUT);
pinMode(STBY,OUTPUT);
comm.begin();
//Serial.begin(9600);
}
void move(int motor, int speed, int direction){
digitalWrite(STBY,HIGH);
boolean inPin1 = HIGH;
boolean inPin2 = LOW;
if (direction == 1){
inPin1 = HIGH;
inPin2 = LOW;
}else{
inPin1 = LOW;
inPin2 = HIGH;

}
if (motor == 1){
digitalWrite(AIM1,inPin1);
digitalWrite(AIM2,inPin2);
analogWrite(PWMA,speed);
}else{
digitalWrite(BIN1,inPin1);
digitalWrite(BIN2,inPin2);
analogWrite(PWMB,speed);
}
}
void loop() {
  int32_t x, y;
  float u, v;
  // Vérifie si un nouveau message de l'Orange PI a été reçu
  if (comm.newMessage() == true) {
    // On lit les 4 valeurs contenues dans le message
    //x = comm.xRead();
    //y = comm.yRead();
    u = comm.uRead();
    v = comm.vRead();
    // et on les renvoie à l'Orange PI
    if (u == 0){
      move(0,0,0);
      move(1,0,0);
    }
    else{
      if(v>0){      
        move(1, (100-v)*0.95, 1);
        move(0, 100+v, 1);}
      else{
        move(1, (100-v)*0.95, 1);
        move(0, 100+v, 1);}
    }
    }
  }

