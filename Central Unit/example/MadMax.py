from state_machine import Car
import logging


logging.info("MadMax initialized, You officialy rock sir !")



def KeyStroke(event) : 
    c = event.char 
    if c == 'a': 
        Car.send(0, 0, 2, 5) 
    elif c == 'd' :
        Car.send(0, 0, 2, -5) 
    elif c == ' ': 
    	root.destroy()
    else : 
        Car.send(0, 0, 2, 0) 



root = tk.Tk()
root.bind_all( "<Key>", KeyStroke)
        #root.withdraw()
root.mainloop() 