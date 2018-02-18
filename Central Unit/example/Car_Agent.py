#inspired by the code @
#https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
import start_sim
import logging
from event import Event
from car import Car 

import tensorflow as tf

from numpy import array
from numpy import argmax

from random import sample
from random import random
from random import randint


AMNT_ACTION = 16 
SAVE_PATH = "./Saves/DriverBrain.ckpt"


def Layer(In_size, Input, Number_of_Neurons, label = None) : 
    W = tf.Variable(tf.random_normal([In_size, Number_of_Neurons]), name = label + "_W")
    B = tf.Variable(tf.random_normal([Number_of_Neurons]), name = label + "_B")
    return tf.matmul(Input, W) + B


def Set_Update_Network(TFVars, Tau) :
    Main_W = [] 
    Target_W = [] 
    for w in TFVars : 
        if "main" in w.name : 
            Main_W.append(w) 
        elif "target" in w.name : 
            Target_W.append(w) 
         
    op_holder = []
    for i in range(len( Main_W) ) : 
        op_holder.append(Target_W[i].assign(Tau * Main_W[i] + (1 - Tau) * Target_W[i]))

    return op_holder

def Update_Network(op_holder, sess) : 
    for op in op_holder : 
        sess.run(op) 

class BabyDriver : 

    def __init__(self, N_shape, StartE, EndE, OffsetE ) : 
        self.e = StartE 
        self.EndE = EndE
        self.OffsetE = OffsetE
        

        self.State_input = tf.placeholder("float", (None, N_shape[0]))
        self.Next_State_input = tf.placeholder("float", ( None, N_shape[0] ) )

        self.Main_Net = self.State_input
        self.Target_Net = self.Next_State_input

        for idx in range( len(N_shape) - 1) : 
            self.Main_Net = Layer(N_shape[idx], self.Main_Net, N_shape[idx + 1], label = "main_" + str(idx) )
            self.Target_Net = Layer(N_shape[idx], self.Main_Net, N_shape[idx + 1], label = "target_" + str(idx) ) 
        
        self.Update_OP = Set_Update_Network(tf.trainable_variables(), 0.001 ) 

    def Move(self, Current_state, Update_E = False) : 
        if random() > self.e : 
            if self.e > self.EndE and Update_E: 
                self.e -= self.OffsetE
            return randint(0, 15)
        else :
            with tf.Session() as sess : 
                sess.run( tf.global_variables_initializer())
                Q_table = sess.run(fetches = [self.Main_Net], feed_dict={self.State_input : Current_state})
                return tf.argmax(Q_table)


    def Train(self, Replay, Y, To_Save_path): 
  
        TQ = self.Target_Net 
        CQ = self.Main_Net 
        
        #1 dimensional array whose length is the amount of batches 
        Target_Q_idx = tf.argmax(TQ, axis = 1 ) 
        Current_Q_idx = tf.argmax(CQ, axis = 1)
        
        Target_Q = tf.one_hot(Target_Q_idx, AMNT_ACTION, axis=1)
        Current_Q = tf.one_hot(Current_Q_idx, AMNT_ACTION, axis = 1 ) 

        #keep only the highest Qs and then reduce to a simple vector ( list )
        TQ = tf.reduce_sum( tf.multiply(TQ, Target_Q), axis = 1  )  
        CQ = tf.reduce_sum( tf.multiply(CQ, Current_Q), axis = 1 ) 
        
        #replay[:, 2] returns a list of the reward for each step in batch 
        loss = tf.reduce_mean( tf.losses.mean_squared_error( Replay[:, 2] + ( Y * TQ), CQ))
        optimizer = tf.train.AdamOptimizer().minimize(loss) 

        with tf.Session() as sess : 
            sess.run(tf.global_variables_initializer())
            sess.run(optimizer, feed_dict = {self.State_input : Replay[:, 0], self.Next_State_input : Replay[:,3]}) 
            Update_Network(self.Update_OP, sess)
            if not (To_Save_path == "" ) : 
                saver = tf.train.Saver() 
                saver.save(sess, To_Save_path) 


class Replay_Memory : 
    def __init__(self, Max_size = 50000) :
        self.buffer = []
        self.max_size = Max_size 
        self.cur_replay = [] 
    
    def add( self, Experience): 
        if len(self.buffer) == self.max_size : 
            self.buffer.pop(0)
        self.buffer.append(Experience) 

    def Sample(self, qty_to_sample) : 
        return array(sample(self.buffer,qty_to_sample))

Schumacher = Car_Agent.BabyDriver([8, 16, 32, 16, 16], 1, 0.1, 0.009)
ReplayBuffer = Car_Agent.Replay_Memory()
total_Step = 0
logging.info( "!!!! In my own script !!!!")


def loop() : 
    #Constants 
    Exploration_Step = 1000
    Y = 0.99
    CheckPt_Step = 10

    global Schumacher
    global ReplayBuffer
    global total_Step 

    event = Event.poll()

    if event.type == Event.PATH : 

        dic, img = event.val
        frame_data = dic.values()
        
        if frame_data[-1] : 
            for i in range( len( frame_data ) - 1) : 
                frame_data[i] = 0 


        if(total_Step < Exploration_Step ): 
            action = randint(0, 15) 
        else : 
            action = Schumacher.Move(frame_data, True)


        reward = 1 
        if frame_data[-1] : 
            reward = -10  


        if ReplayBuffer.cur_replay == [] : 
            ReplayBuffer.cur_replay.append(frame_data) 
            ReplayBuffer.cur_replay.append(action) 
            ReplayBuffer.cur_replay.append(reward) 
        else : 
            ReplayBuffer.cur_replay.append(frame_data) 
            ReplayBuffer.add(ReplayBuffer.cur_replay) 
            ReplayBuffer.cur_replay = [frame_data, action, reward]

        if total_Step >= Exploration_Step : 
            if total_Step % CheckPt_Step == 0 :
                Schumacher.Train(ReplayBuffer.Sample(250), Y, SAVE_PATH) 
            else : 
                Schumacher.Train(ReplayBuffer.Sample(250), Y, "") 

        total_Step += 1

        if frame_data[-1] : 
            start_sim.subprocs[-1].kill()
            start_sim.subprocs.append(subprocess.Popen([start_sim.PYTHON2, '../bin/simulator.py', '--arena', start_sim.args.arena]))
            total_Step = 0 
        pass