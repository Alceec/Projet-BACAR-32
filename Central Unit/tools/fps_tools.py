"""
Frame Per Second 
generator that help to count the FPS
"""
from time import time,sleep
from collections import deque

def fps_generator(maxlen=10):
    """
    generator
    > fps = fps_generator(10)
    ... in the loop
        print(fps.next())        
    """
    d = deque(maxlen=maxlen)
    for i in range(maxlen):
        d.append(time())
    while True:
        t = time()
        d.append(t)
        yield (maxlen-1)/(t-d[0])
        
        
if __name__ == '__main__':
    
    fps = fps_generator(5)
    for i in range(20):
        print(fps.next())
        sleep(.1)        
