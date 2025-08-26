import math
import numpy as np
import tensorflow as tf
import os
import observationHistory
NN_MODEL_Actor1 = './pb_model/5-5.pb'
NN_MODEL_Actor1_CKPT = '/home/bupt/PPO_2/PPOServer_socreDDPG_connect_V1/model/16000mmmodel.ckpt-16000'
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn
import random
from utils import A_DIM,DELTA_VECTOR
FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.90
EPS = 0.2
S_LEN = 4
Input_LEN = 2

class Network():

    def CreateNetwork(self, inputs):  # actor2-critic

        with tf.variable_scope('actor2-critic'):
            value = tflearn.fully_connected(self.CreateCore(inputs), 1, activation='linear')
            return value

    def Createpi(self,inputs):  # actor1 + actor2
        with tf.variable_scope('Actor'):
            net1 = tflearn.fully_connected(inputs, n_units=64, activation='tanh')
            # net = tflearn.flatten(net)
            net2 = tflearn.fully_connected(net1, n_units=32,activation='tanh')
            net3 = tflearn.fully_connected(net2, n_units=A_DIM, activation='linear')
            logits = tflearn.fully_connected(net3, A_DIM, activation='softmax')
            return logits, net1, net2, net3
    # def Createpi_noconcet(sel,inputs):
    #     net = tflearn.fully_connected(inputs, n_units=64, activation='tanh')
    #     net = tflearn.fully_connected(net,n_units=32,activation='tanh')
    #     net = tflearn.fully_connected(net,n_units=15,activation='softmax')
    #     return net
    def CreateCore(self, inputs):
        inputs = tflearn.flatten(inputs)
        inputs = tflearn.fully_connected(inputs, n_units=64, activation='tanh')
        net = tflearn.fully_connected(inputs, n_units=32, activation='tanh')
        return net

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })



    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.loss_history = []
        
        self.training_epo = 10
        self.s_dim = state_dim
        self.a_dim = action_dim # actor2
        self.lr_rate = learning_rate
        self.sess = sess
        

        self.R = tf.placeholder(tf.float32, [None, 1])
        # input_ckpt是加载的ckpt模型的输入
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim], name='x') #actor2

        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        self.entropy_weight = tf.placeholder(tf.float32)
        # pi是加载ckpt模型继续训练中的输出的策略
        self.pi, self.net1, self.net2, self.net3 = self.Createpi(inputs=self.inputs) # actor1 + actor2 policy result

        self.val= self.CreateNetwork(inputs=self.inputs)

        self.real_out_actor2 = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = tf.multiply(self.real_out_actor2, tf.log(self.real_out_actor2))
        self.adv = self.R - tf.stop_gradient(self.val)

        self.ratio = tf.reduce_sum(tf.multiply(self.real_out_actor2, self.acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(self.old_pi, self.acts), reduction_indices=1, keepdims=True)

        self.ppo2loss = tf.minimum(self.ratio * self.adv,   # actor2
                            tf.clip_by_value(self.ratio, 1 - EPS, 1 + EPS) * self.adv
                        )

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        self.loss = -tf.reduce_sum(self.ppo2loss) \
            + self.entropy_weight * tf.reduce_sum(self.entropy)

        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
        self.val_loss = 0.5 * tflearn.mean_square(self.val, self.R)
        self.val_optimize = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss)

    def getppo2loss(self,input):
        loss = self.sess.run(self.ppo2loss,feed_dict = {
            self.inputs:input
        })
        return loss
    
    def predict(self, input): # actor2
        action = self.sess.run(self.real_out_actor2, feed_dict={
            self.inputs: input
        })
        return action[0]
    def getlinear(self, input):
        net1= self.sess.run(self.net1, feed_dict={
            self.inputs: input
        })
        return net1
    def getlinear2(self, input):
        net2= self.sess.run(self.net2, feed_dict={
            self.inputs: input
        })
        return net2
    def getlinear3(self, input):
        net3= self.sess.run(self.net3, feed_dict={
            self.inputs: input
        })
        return net3
    def getvalue(self,input):
        value = self.sess.run(self.val,feed_dict={self.inputs:input})
        #print('value:',value)
        return value

    def get_entropy(self):
        return 0.1
    def set_prob_gcc(self,prob_gcc):
        self.prob_GCC = prob_gcc

    def train(self, actor1_batch ,s_batch, a_batch, p_batch, v_batch, epoch, batch_size = 64):
        # shuffle is all you need
        s_batch, a_batch, p_batch, v_batch = \
            tflearn.data_utils.shuffle(s_batch, a_batch, p_batch, v_batch)
        # mini_batch
        i, train_len = 0, s_batch.shape[0]
        while train_len > 0:

            _batch_size = np.minimum(batch_size, train_len)
            self.sess.run([self.optimize, self.val_optimize], feed_dict={
                self.inputs: s_batch[i:i+_batch_size],
                self.acts: a_batch[i:i+_batch_size],
                self.R: v_batch[i:i+_batch_size],
                self.old_pi: p_batch[i:i+_batch_size],

                self.entropy_weight: self.get_entropy()
            })
            train_len -= _batch_size
            i += _batch_size
        
        

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        v_batch = self.sess.run(self.val, feed_dict={
            self.inputs: s_batch
        })
        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        td_batch = R_batch - v_batch
        return list(R_batch), list(td_batch)

    def compute_gae_v(self, s_batch, a_batch, r_batch, terminal, GAE_LAMBDA = 0.95):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])
        mb_advs = np.zeros_like(R_batch)
        lastgaelam = 0.

        v_batch = self.sess.run(self.val, feed_dict={
            self.inputs: s_batch
        })
        if terminal:
            v_batch[-1, 0] = 0  # terminal state

        for t in reversed(range(ba_size - 1)):
            delta = r_batch[t] + GAMMA * v_batch[t+1] - v_batch[t]
            mb_advs[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * lastgaelam
        R_batch = mb_advs + v_batch
        return list(R_batch), list(mb_advs)

    def get_variable(self):
            pbvar = self.sess.graph.get_tensor_by_name('FullyConnected/W:0')
            ckptvar = self.sess.graph.get_tensor_by_name('FullyConnected/W/Adam_1:0')

class BlackboxGCC(object):
    def __init__(self,sess):
        self.sess = sess


        self.delta = DELTA_VECTOR
        # self.delta = [0.7, 0.8, 0.98, 1.0025, 1.005, 1.07]
        self.load_pb()
        # self.logits = self.NN4(self.x1)
        self.input_x = self.sess.graph.get_tensor_by_name('x_ckpt:0')
        self.op = self.sess.graph.get_tensor_by_name('output:0')   # black-box GCC output
        self.layer = self.sess.graph.get_tensor_by_name('fc_128/Reshape:0')  # featuremap
        self.gcc_action = tf.cast(tf.argmax(self.op, 1), tf.int32)
        self.gcc1 = self.gcc_action[0]
    def load_pb(self):
        with tf.gfile.GFile(NN_MODEL_Actor1, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def get_bitrate(self,input,use_bitrate):
        predict = self.gcc1.eval(feed_dict={self.input_x: input})
        use_bitrate = self.delta[int(predict)] * use_bitrate
        return use_bitrate
    def get_prob(self,input):
        predict = self.op.eval(feed_dict={self.input_x: input})[0]

        return predict
    def getlayer(self):
        return self.layer
    def getinput(self):
        return self.input_x