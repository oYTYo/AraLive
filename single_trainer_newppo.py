import numpy as np
import tensorflow as tf
import tflearn
import gym
import os
import datetime
import utils
import time
import _thread
import scipy.io as sio
import logging
from gym import spaces
from gym.utils import seeding
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import ppo_score as network
import socket
from observationHistory import ObservationHistory
from utils import current_time, StopException, log_time, rl_log
from utils import STATE_MIN, STATE_MAX,A_DIM
from utils import ACTOR_VECTOR, LEARN_STEP,DELTA_VECTOR,CKPT_DIR
import sys

S_LEN = 4
Input_LEN = 8
S_DIM = S_LEN*Input_LEN
ACTOR2_LEN = 4
BATCH_SIZE = 32
ACTOR_LR_RATE = 2.5e-3  #1e-4
TRAIN_SEQ_LEN = 128  # take as a train batch
TRAIN_STEPS = int(2e8)
RANDOM_SEED = 42
SUMMARY_DIR = './results/ppo'
MODEL_SAVE_INTERVAL = 100
s = socket.socket()
g1 = tf.Graph()
arr = []
brr = []
bitrate = sio.loadmat("Query 1701-1800_channel514.mat")['bw'][0]

class OnlineEnvironment(gym.Env):
    USE_GAP = False

    def __init__(self):
        self.rand = None
        self.seed = None
        self.reward_range = None
        self.action_space = spaces.Discrete(n=13)
        self.reward = 0.0
        self.init_flag = False
        self.observationHistory = ObservationHistory()
        self.observation_space = spaces.Box(low=STATE_MIN, high=STATE_MAX, dtype=np.float32)
        self.observation = STATE_MIN
        self.current_action = 1e6
        self.last_action = 1e6
        self.pipe = None
        self.gcc = 0
        self.ppo = 0
        self.cc = 0
        self.gcc_bitrate = 0

        # 记载log
        self.f = None
        self.f_count = None
        self.f_gcc_or_ppo = None

        self.is_open_gcc = True

        self.start_time = datetime.datetime.now()
        self.current_time = datetime.datetime.now()

        self.first_x_minute = 0  # 开始一分钟使用gcc
        self.overuse_flag = 0
        self.start = 0
        self.step_flag = False
        self.load_flag = False
    def step(self, action):
        done = False
        info = {}

        try:
            data = self.pipe.recv_bytes(1024)
            data = data.decode('utf-8')
        except BrokenPipeError:
            self.close()
            raise StopException("STOP Exception")


        if data[0:4] == "STOP":
            self.f_gcc_or_ppo.write(current_time() + ' stop training\n')
            self.close()
            raise StopException("STOP Exception")
        else:
            self.reward = 0
            # 用来判断是否启用RL的估计值
            try:
                packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate = self.observationHistory.decode_data(
                    RTCPinfo=data)
            except (Exception, ValueError):
                rl_log(flag="ERROR", content="Decode Data Error")
                self.close()
                raise StopException("STOP Exception")
            send_data = '1'
            #  设置obseervation中的last_action
            self.observationHistory.set_last_action(last_action=action)
            #  接受并分析webrtc传入数据
            self.observationHistory.update_online_observation(RTCPinfo=data, t=time.time())
            # 得到当前的action并传回webrtc端

            # obs_input = np.zeros([1, Input_LEN, Input_LEN])
            # obs_input[0:1, :, :] = self.observation
            self.current_action = int(action)
            send_data += str(self.current_action)
            send_data += (1024 - len(send_data)) * '%'

            self.cc += 1
            # if self.cc % 100 == 0:
            #     # 每100次记录下gcc和ppo的次数
            # self.f_count.write(current_time() + ' gcc vs ppo ' + str(self.gcc) + ' ' + str(self.ppo) + '\n')
            # self.f_count.flush()

            # 将当前码率发回线程端
            self.pipe.send_bytes(send_data.encode('utf-8'), 0, 1024)

            if OnlineEnvironment.USE_GAP:
                self.observation = self.observationHistory.as_array_with_pacing()
            else:
                self.observation = self.observationHistory.as_array()
            # 计算reward
            # self.reward = self.observationHistory.get_new_humanreward()
            self.reward = self.observationHistory.get_new_humanreward()
            # 存储上一次的action

            self.last_action = self.current_action

            return self.observation, self.reward, done, info

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]
    def get_gccbitrate(self):
        return self.gcc_bitrate
    def render(self, mode='human'):
        pass

    def reset(self):
        return self.observation

    def add_pipe(self, pipe):
        self.pipe = pipe
    def add_dir(self, directory):
        cur = log_time()  # 获取当前日期作为日志名
        self.observationHistory.log_file = open(directory + '/' + cur + '_log.txt', 'a+')
        self.f_gcc_or_ppo = open(directory + '/' + cur + '_gcc_or_ppo.txt', 'a+')

    def set_first_minute(self, x_minute):
        self.first_x_minute = x_minute

    @staticmethod
    def set_use_gap(flag):
        OnlineEnvironment.USE_GAP = flag

    def set_continue_gcc_times(self, count):
        self.continue_gcc_times = count
    def set_reward(self, throughput_param, delay_param, loss_param, smooth_param):
        self.observationHistory.set_parameters(throughput_param, delay_param, loss_param, smooth_param)

    def close(self):
        self.f_gcc_or_ppo.flush()
        self.f_gcc_or_ppo.close()
    def get_overuseflag(self):
        return self.overuse_flag
    def rec_data(self):
        try:
            data = self.pipe.recv_bytes(1024)
            data = data.decode('utf-8')
        except BrokenPipeError:
            self.close()
            raise StopException("STOP Exception")

        if data[0:4] == "STOP":
            self.f_gcc_or_ppo.write(current_time() + ' stop training\n')
            self.close()
            raise StopException("STOP Exception")
        else:
            self.reward = 0
            # 用来判断是否启用RL的估计值
            try:
                packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate = self.observationHistory.decode_data(
                    RTCPinfo=data)
            except (Exception, ValueError):
                rl_log(flag="ERROR", content="Decode Data Error")
                self.close()
                raise StopException("STOP Exception")


class Trainer(object):

    def __init__(self, pipe, directory):
        self.env = DummyVecEnv([lambda: gym.make("OnlineEnv-v0")])

        self.user_id = directory[0:directory.find('_')]  # 只有id
        self.make_dir(self.user_id)
        self.directory = directory  # 这里的directory就是user_mark，包含id,isp,net,time
        self.pipe = pipe
        self.add_pipe(pipe=pipe)
        self.add_dir(directory='users/' + self.user_id)
        self.use_memory = True
        self.one_time_steps = 1024
        self.score = 0.5
        self.choose_property = True
        self.choose_property = True
        self.observation = ObservationHistory
        self.load_flag = True
    def build_summaries(self):
        td_loss = tf.Variable(0.)
        tf.summary.scalar("TD_loss", td_loss)
        eps_total_reward = tf.Variable(0.)
        tf.summary.scalar("Eps_total_reward", eps_total_reward)

        summary_vars = [td_loss, eps_total_reward]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def run_actor(self):
        env = self.env

        with tf.Session() as sess1:
            actor = network.Network(sess1,
                                    state_dim=S_DIM, action_dim=A_DIM,
                                    learning_rate=ACTOR_LR_RATE)
            blackbox = network.BlackboxGCC(sess=sess1)

            summary_ops, summary_vars = self.build_summaries()
            sess1.run(tf.global_variables_initializer())
            saver = tf.train.Saver()  # save neural net parameters
            fake_epochs = 0
            obs = env.reset()
            obs = np.reshape(obs, (4, 8))
            
            variable_names = [v.name for v in tf.all_variables()]
            actor_2_input = obs
            actor1_batch, s_batch, a_batch, p_batch, r_batch, x_batch = [], [], [], [], [],[]
            blackbox_bitrate = 1000000
            hybrid_bitrate = 1000000
            start = time.time()
            step = 0
            f = open("Logs/bitrate.txt", "a")
            while(True):
                # for step in range(TRAIN_SEQ_LEN):
                # 一个batch中的观察值
                ###########  time ###########
                now = time.time()

                actor1_input = np.zeros([1, 8])
                actor1_input[0][:4] = np.reshape(obs, (S_LEN * Input_LEN,))[4:8] * 50
                actor1_input[0][4:] = np.reshape(obs, (S_LEN * Input_LEN,))[20:24]
                actor_2_input = np.reshape(actor_2_input, (1,S_DIM))
                print('actor_2_input:',actor_2_input)
                s_batch.append(np.reshape(actor_2_input, S_DIM))
                actor1_batch.append(np.reshape(actor1_input, (8,)))
                # softmax输出的概率预测值
                actor2_prob = actor.predict(actor_2_input)
                blackbox_prob = blackbox.get_prob(actor1_input)
                # print('linear1',actor.getlinear(actor_2_input))
                # print('linear2',actor.getlinear2(actor_2_input))
                # print('linear3',actor.getlinear3(actor_2_input))

                print('blackbox_prob:',blackbox_prob)
                print('index:',np.argmax(blackbox_prob))
                #print('loss:',np.mean(actor1_input[0][:4]))
                if(np.argmax(blackbox_prob)<=4):
                    print('overshot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    blackbox_prob = blackbox_prob*20
                    blackbox_prob = np.power(3, blackbox_prob)
                    blackbox_prob_attention = blackbox_prob
                    score_action = 0.25  # 0.25
                else: 
                    blackbox_prob_attention = self.sigmoid(blackbox_prob)
                    score_action = 0.5  # 0.5
                if (step % 20 == 0):
                    blackbox_bitrate = blackbox.get_bitrate(actor1_input, hybrid_bitrate)
                else:
                    blackbox_bitrate = blackbox.get_bitrate(actor1_input, blackbox_bitrate)
                actor2_prob = actor2_prob*blackbox_prob_attention
                # print('black_box_attentio:',blackbox_prob_attention)
                # print('actor2_prob:',actor2_prob)
                actor2_prob = actor2_prob / actor2_prob.sum()
                # print('sum',actor2_prob.sum())
                try:
                    actor2_action = np.random.choice(A_DIM, p=np.nan_to_num(actor2_prob))
                except:
                     print('sum no equal 1')
                     actor2_action = np.argmax(actor2_prob)

                actor2_bitrate = int(ACTOR_VECTOR[actor2_action]*1e6)
                hybrid_bitrate = int((score_action * actor2_bitrate) + (1 - score_action) * blackbox_bitrate)
                hybrid_bitrate = self.lim_bitrate(hybrid_bitrate)
                # print('black_prob_attention:', blackbox_prob_attention)
                # print('actor2_prob:', actor2_prob)
                f.write(str(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')[:-2])+" "+ " " + str(int(blackbox_bitrate)) + " " + str(int(actor2_bitrate))+" "+str(int(hybrid_bitrate))+" "+ '\n')
                
                
                obs, rew, done, info = env.step([int(hybrid_bitrate)])

                #################################################
                # save reward [__]
                t = (int(now-start))                
                brr.append(hybrid_bitrate)
                brr.append(max(bitrate[t+6]-100000,bitrate[t+6]*0.85))
                brr.append(rew)
                
                ###############################
                # save observations [__time__][ __,__,__,__,]  
                time_ = now - start
                # time_ = time_.seconds+time_.microseconds*1000
                print("time",time_)
           
                # print("time.micrsecnds".time.micrseconds)
                arr.append(time_)
                arr.append(np.reshape(actor_2_input, (S_DIM,)))
                arr.append(hybrid_bitrate)
                

                ###############################

                obs = np.reshape(obs, (S_LEN, Input_LEN))
                actor_2_input = obs
                # save mem
                action_vec = np.zeros(A_DIM)
                action_vec[actor2_action] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(actor2_prob)
                step = step + 1
                print('actor2_prob:',actor2_prob)      
                print('actor2_bitrate: ' + str(actor2_bitrate))
                print('blackbox_bitrate: ' + str(blackbox_bitrate))
                print('hybrid_action:' + str(hybrid_bitrate))
                print('blackbox_action:  ' + str(np.argmax(blackbox_prob)))
                print('step: '+str(step))

                print("time:",t)
                if t % 2000 == 0:

                        np.save('state_-'+str(t)+'.npy',np.array(arr))
                        np.save('reward_-'+str(t)+'.npy',np.array(brr))
                        
                if done:
                    break
                if len(r_batch) >= TRAIN_SEQ_LEN:  # batch_size, and report experience to the coordinator
                    # f2.write(str(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')[:-2]) + " " + str(ep_reward)+'\n')
                    ep_reward = 0
                    total_reward, total_td_loss = [], []

                    v_, td_ = actor.compute_v(s_batch, a_batch, r_batch, done)

                    total_reward.append(np.sum(v_))
                    total_td_loss.append(np.sum(td_))
                    
                    s_batch = np.stack(s_batch, axis=0)
                    actor1_batch = np.stack(actor1_batch,axis=0)
                    
                    a_batch = np.vstack(a_batch)
                    p_batch = np.vstack(p_batch)
                    v_batch = np.vstack(r_batch)

                    for _ in range(actor.training_epo):
                        # var = var * 0.98
                        actor.train(actor1_batch,s_batch, a_batch, p_batch, v_batch,step)

                    summary_str = sess1.run(summary_ops, feed_dict={
                        summary_vars[0]: np.mean(total_td_loss),
                        summary_vars[1]: np.mean(total_reward)
                    })
                    fake_epochs += 1
                    print("fake_epochs:",fake_epochs)
                    

                    actor1_batch,s_batch, a_batch, p_batch, r_batch, x_batch = [], [], [], [], [], []
                    if fake_epochs % MODEL_SAVE_INTERVAL == 0:

                        np.save('state_'+str(t)+'.npy',np.array(arr))
                        np.save('reward_'+str(t)+'.npy',np.array(brr))

                    
                        

    # 是否需要选择相同属性
    def set_choose_property(self, flag):
        self.choose_property = flag
    def sigmoid(self,input):
        s = 1/(1+np.exp(-input))
        return s
    def lim_bitrate(self,bitrate):
        if (bitrate > 2000000):
            bitrate = 2000000
        elif (bitrate < 300000):
            bitrate = 300000
        return bitrate
    @staticmethod
    def make_dir(path):
        if not os.path.exists('users/' + path):
            os.makedirs('users/' + path)

    def load_model(self, user_mark):
        user_id = user_mark[0:user_mark.find('_')]
        # rl_log(flag="INFO", content='The user id is ' + user_id + '.')

        if user_id == "default":
            return False
        default_property = user_mark[user_mark.find('_') + 1:]
        user_directory = 'users/'
        user_list = os.listdir(user_directory)

        # 获取融合后的模型的时间
        default_model = 'users/default/PPO2.zip'
        default_model_modify_time = 0
        if os.path.exists(default_model):
            default_model_modify_time = os.path.getmtime(default_model)

        # 存在该用户模型
        for user_item in user_list:
            if user_item.find(user_id) != -1:
                user_directory = "users/" + user_item
                if self.is_model_exist(user_directory):
                    # 用户模型比融合模型的时间新
                    if os.path.getmtime(user_directory + '/PPO2.zip') > default_model_modify_time:
                        self.model = PPO2.load(load_path=user_directory + '/PPO2', env=self.env)
                        return True
        # 使用默认
        if self.choose_property:
            user_directory = "users/default"
            if self.is_model_exist(user_directory):
                self.model = PPO2.load(load_path=user_directory + '/PPO2', env=self.env)
                return True
        return False

    def run(self):
        start_flag = 0
        while True:
            # 只有刚开播第一次进入的时候，才判断需不需要加载模型
            if start_flag == 0:
                if self.load_model(self.directory):
                    rl_log(flag="INFO", user_id=str(self.user_id), content="load the old model.")
                else:
                    rl_log(flag="INFO", user_id=str(self.user_id), content="start new training.")

            try:
                start_flag += 1
                rl_log(flag="INFO", user_id=str(self.user_id), content="learning process " + str(start_flag) + ".")
                # 开始训练
                self.model.learn(total_timesteps=self.one_time_steps)
                # self.save(path=self.user_id + '/PPO2_'+str(start_flag))
                self.save(path=self.user_id + '/PPO2')
            except StopException:
                # self.save(path=self.user_id + '/PPO2')
                self.pipe.send("KILL")
                self.pipe.close()
                break
    def save(self, path):
        self.model.save(save_path='users/' + path, cloudpickle=False)
        rl_log(flag="INFO", user_id=self.user_id, content="save the model.")

    def add_pipe(self, pipe):
        self.env.env_method("add_pipe", pipe)

    def add_dir(self, directory):

        self.env.env_method("add_dir",directory)
    def set_use_gap(self, flag):
        self.env.env_method("set_use_gap", flag)

    def set_first_x_minute(self, x_minute):
        self.env.env_method("set_first_minute", x_minute)

    def set_time_steps(self, total_steps):
        self.one_time_steps = total_steps

    def set_continue_gcc(self, count):
        self.env.env_method("set_continue_gcc_times", count)


    def set_reward_parameters(self, throughput_param, delay_param, loss_param, smooth_param):
        self.env.env_method("set_reward", throughput_param, delay_param, loss_param, smooth_param)
    def get_gccbitrate(self):
        return self.env.env_method("get_gccbitrate")
    def get_step(self):
        return  self.env.env_method("get_step")
    def get_overuseflag(self):
        return self.env.env_method("get_overuseflag")
