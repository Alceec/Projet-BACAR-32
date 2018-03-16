import socket
import pygame

client = socket.socket()
address = ('192.168.42.1', 5555) 
print('waiting for connection') 
client.connect(address) 
print('connected') 

pygame.init()
screen = pygame.display.set_mode((100, 100))

#main loop 
while True : 
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key <= 256:
            client.send(str.encode(chr(event.key)))