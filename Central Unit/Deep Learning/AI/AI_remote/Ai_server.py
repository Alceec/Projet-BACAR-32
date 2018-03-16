import socket
from keras.models import load_model
import numpy as np

HOST = ''
PORT = 5555
sock = socket.socket()
address = (HOST, PORT) 
print(socket.gethostname()) 
sock.bind(address) 
sock.listen(1) 
print("waiting for connection") 
conn, add = sock.accept()
print('connection established')
model = load_model('/home/student/Desktop/bacar_distrib/Central Unit/la-voiture-qui-dechire-tout-sa-maman-MkII/Central Unit/example/NN_data/K2000.h5')

def Predict(data) : 

    global model 

    command = model.predict(data)
    #logging.info('\n\n {} \n\n'.format(command[0]))
    command = np.argmax(command[0])

 
    if command == 0 : 
        button = 'a' 
    elif command == 1 : 
        button = 'w' 
    elif command == 2 : 
        button = 'd' 

    return button 
 

#main loop 
while True :
	data = conn.recv( 1024 )
	mem = np.array( [eval(data.decode())] ) 
 
	button = Predict(mem / 150) 
	conn.send(button.encode())