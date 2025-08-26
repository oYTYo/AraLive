import numpy as np
import time
from utils import current_time
from utils import LEARN_STEP, LEARN_STEP_PACKET
from humanrewardmodel import NeuralNet
import tensorflow as tf

import torch
from Mamba2 import MambaBlock
import concurrent.futures
import os

HISTORY_LEN = 8
SUM = 0

class ObservationHistory(object):
    def __init__(self):
        # average
        self.learning_step = LEARN_STEP
        self.throughput = 0.0
        self.loss = 0.0
        self.delay = 0.0
        self.delay_interval = 0.0
        self.min_delay = 0.0
        # 4*4
        self.loss_window = HISTORY_LEN * [0]
        self.delay_window = HISTORY_LEN * [0]
        self.throughput_window = HISTORY_LEN * [0]
        self.delay_interval_window = HISTORY_LEN * [0]

        self.timeseries = []
        self.loss_movewindow = []
        self.delay_movewindow = []
        self.throughput_movewindow = []
        self.delay_interval_movewindow = []

        self.gap_window = HISTORY_LEN * [0]
        self.last_action = 0
        # 120*4
        self.loss_window_step = []
        self.delay_window_step = []
        self.throughput_window_step = []
        self.delay_interval_window_step = []
        # use history info
        self.history_buffer = []
        # concerto
        self.loss_concerto = 60 * [1.0]
        self.delay_interval_concerto = 60 * [0.0]
        self.throughput_concerto = 15 * [0.0]

        self.log_file = None

        self.reward = 0.0
        self.count = 0
        self.buffer = []
        self.throughput_parameter = 50
        self.delay_parameter = 10.0
        self.loss_parameter = 50
        self.smooth_parameter = 30
        self.buffer_loss = []
        self.buffer_throughput = []
        self.buffer_delay = []
        self.buffer_delay_interval = []
        self.buffer_gap = []
        self.step = 0
        self.step_packet_length = 0

        self.reward_list = [0] * 5
        self.boost = False

        # # 初始化 TensorFlow Lite 模型
        # self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        # self.interpreter.allocate_tensors()
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()

        # 记录一下每次的reward
        self.log_file1 = './reward.log'
        # 检查日志文件是否存在，如果不存在则写入表头
        if not os.path.exists(self.log_file1):
            with open(self.log_file1, 'w') as f:
                f.write("Time, Result1, Result2, Self_Reward, Self_Boost\n")


    def set_parameters(self, throughput_param, delay_param, loss_param, smooth_param):
        self.throughput_parameter = throughput_param
        self.delay_parameter = delay_param
        self.loss_parameter = loss_param
        self.smooth_parameter = smooth_param
        print(f"[DEBUG] Parameters set: throughput={throughput_param}, delay={delay_param}, loss={loss_param}, smooth={smooth_param}")

    def get_parameters(self):
        params = f"th:{self.throughput_parameter} delay:{self.delay_parameter} loss:{self.loss_parameter} smooth:{self.smooth_parameter}"
        print(f"[DEBUG] Current parameters: {params}")
        return params

    """ 正常tf模型 """

    def get_humanreward(self,t):
        # 记录推理开始时间
        start_time = time.time()
        # 创建默认图
        with tf.Graph().as_default():
            # 使用会话加载模型
            with tf.compat.v1.Session() as sess:
                # 创建模型实例
                neural_net = NeuralNet()
                # 加载权重
                neural_net.load_weights(filepath="./model/tfmodel.ckpt")

                # 获取输入数据
                batch_x = self.get_200ms_state(t=t)
                state = neural_net.transform(batch_x)  # 假设 transform 方法直接处理 NumPy 数据

                # 在会话中运行前向传播
                pred = sess.run(neural_net(state)) # 将 state 输入到模型中并获取结果

            # # 计算奖励
            # self.reward = np.argmax(pred) / 2

            reward = np.argmax(pred) / 2
            # 输出推理结果
            print(f"AraLive reward: {reward}")

            # print(f"Human reward / 2 ===============: {self.reward}")

            # 记录推理结束时间
            end_time = time.time()
            # 输出推理时间
            inference_time = end_time - start_time
            print(f"Get_Human_Reward time: {inference_time:.4f} seconds")

            # return self.reward
            return reward


    def get_YTYreward(self,t):
        # 记录推理开始时间
        start_time = time.time()

        model = MambaBlock()
        model.load_state_dict(torch.load('mamba_01_best.pth', map_location=torch.device('cpu')))
        model.eval()
        batch_x = np.array(self.get_200ms_state(t=t))
        input = torch.from_numpy(batch_x.reshape(1,3,10))
        input = input.swapaxes(1, 2)
        yty = model(input.float())
        probabilities = torch.sigmoid(model.classifier(yty))
        # 输出推理结果
        print(f" YTY Probabilities: {probabilities}")


        # 记录推理结束时间
        end_time = time.time()
        # 输出推理时间
        inference_time = end_time - start_time
        print(f"Get_YTY_Reward time: {inference_time:.4f} seconds")

        
        self.reward_list.pop(0)
        self.reward_list.append(probabilities.item())
        mean_reward = sum(self.reward_list) / len(self.reward_list)
        if mean_reward > 0.5 :
            self.boost = True
        elif mean_reward <= 0.5 :
            self.boost = False

        return probabilities.item()


    def get_new_humanreward(self):
        t = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 同时运行两个模型推理
            future1 = executor.submit(self.get_humanreward, t)
            future2 = executor.submit(self.get_YTYreward, t)

            # 等待推理完成并获取结果
            result1 = future1.result()
            result2 = future2.result()

            
            if not self.boost :
                if result1 <= 1.5 and result2 <= 0.03233:
                    self.reward = 1
                elif result1 <= 2.5 and result2 <= 0.034 :
                    self.reward = 2
                elif result1 <= 3.5 and result2 <= 0.04 :
                    self.reward = 3
                elif result1 == 4.5 and result2 <= 0.1 :
                    self.reward = 4
                else :
                    self.reward = result1

            else :
                if result1 == 4.5:
                    self.reward = 5
                elif result1 >= 3.5:
                    self.reward = 4 
                elif result1 >= 2.5:
                    self.reward = 3
                elif result1 >= 1.5:
                    self.reward = 2
                else :
                    self.reward = 1

            # 准备日志内容
            log_entry = f"{t}, {result1}, {result2}, {self.reward}, {self.boost}\n"

            # 写入日志文件
            with open(self.log_file1, 'a') as f:
                f.write(log_entry)

            return self.reward

    """ lite版本 """
    # def get_humanreward(self):
    #     print("[DEBUG] Starting human reward computation.")
    #     start_time = time.time()  # 记录起始时间
    #
    #     # 获取输入
    #     batch_x = self.get_200ms_state(t=time.time())
    #     if all(v == 0 for v in batch_x):
    #         print(f"[WARNING] Input batch_x contains mostly zeros: {batch_x}")
    #
    #     # 模型预测
    #     try:
    #         # 设置 TensorFlow Lite 输入
    #         self.interpreter.set_tensor(self.input_details[0]['index'],
    #                                     np.array(batch_x, dtype=np.float32).reshape(1, -1))
    #
    #         # 执行推理
    #         self.interpreter.invoke()
    #
    #         # 获取输出
    #         output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
    #         print(f"[DEBUG] Prediction output: {output_data}")
    #         self.reward = np.argmax(output_data) / 2
    #
    #     except Exception as e:
    #         print(f"[ERROR] TensorFlow Lite inference error: {e}")
    #         self.reward = -1  # Return an indicative error reward
    #
    #     end_time = time.time()  # 记录结束时间
    #     elapsed_time = end_time - start_time  # 计算耗时
    #     print(f"[DEBUG] Time for prediction: {elapsed_time:.4f} seconds")
    #
    #     print(f"[DEBUG] Computed reward: {self.reward}")
    #     return self.reward



    def get_reward(self):

        b = self.delay_parameter
        # if flag == 1:
        #     # 过载的情况，需要惩罚
        #     b += np.mean(self.delay_window) / 10.0

        self.reward = self.throughput_parameter * np.mean(self.throughput_window) - b * np.mean(
            self.delay_window) - self.loss_parameter * np.mean(
            self.loss_window)
        return self.reward

    # 2020/9/16 start use new type of reward
    def get_reward_mean_policy(self,last_action, curret_action):
        if (np.mean(self.delay_window) <= 2):
            delay_use = 1
        else:
            delay_use = np.mean(self.delay_window)
        throughput = np.mean(self.throughput_window)
        loss = np.mean(self.loss_window)
        self.reward = 10 * ((throughput - 2*loss)/delay_use - 0.15*abs(last_action-curret_action)/1e6)

        return self.reward

    def mean_nozero(self, obj):
        sum = 0
        length = 0

        for i in obj:
            if i != 0:
                sum += i
                length += 1
        if(length != 0):
            mean_nozero = sum / length
        else:
            mean_nozero = 0

        return mean_nozero

    def get_reward_stepinfo_policy(self):
        if (self.mean_nozero(self.delay_window_step)) <= 1.8:
            delay_use = 1.8
        else:
            delay_use = self.mean_nozero(self.delay_window_step)
        self.reward = (self.mean_nozero(self.throughput_window_step) - self.mean_nozero(
            self.loss_window_step)) / delay_use
        return self.reward

    def set_last_action(self, last_action):
        self.last_action = last_action

    def log(self, action):
        # dt_ms = datetime.datetime.now().strftime('%H:%M:%S:%f')
        info = current_time() + ' ' + str(round(action, 2)) + ' ' + str(round(self.throughput, 2)) + ' ' + str(
            round(self.loss, 2)) + ' ' + str(round(self.delay, 2)) + ' ' + str(round(self.reward, 2)) + '\n'
        self.log_file.write(info)
        self.count += 1
        if self.count % 1000:
            self.log_file.flush()

    @staticmethod
    def decode_data(RTCPinfo):
        throughput_length = []
        delay_interval_length = []
        throughput = []
        delay_interval = []

        packet_length = int(ord(RTCPinfo[0]) - 48)
        for i in range(packet_length):
            throughput_length.append(int(ord(RTCPinfo[i + 1]) - 48))
            delay_interval_length.append(int(ord(RTCPinfo[i + 1 + packet_length]) - 48))
        loss_length = int(ord(RTCPinfo[2 * packet_length + 1]) - 48)
        rtt_length = int(ord(RTCPinfo[2 * packet_length + 2]) - 48)
        pacing_bitrate_length = int(ord(RTCPinfo[2 * packet_length + 3]) - 48)
        start = 1 + 2 * packet_length + 3

        for i in range(packet_length):
            throughput.append(float(RTCPinfo[start: start + throughput_length[i]]) / 2)
            start += throughput_length[i]
        for i in range(packet_length):
            delay_interval.append(float(RTCPinfo[start: start + delay_interval_length[i]]))
            start += delay_interval_length[i]
        loss = float(RTCPinfo[start: start + loss_length]) / 100
        start += loss_length
        rtt = int(RTCPinfo[start: start + rtt_length])
        start += rtt_length
        pacing_bitrate = int(RTCPinfo[start: start + pacing_bitrate_length])

        return packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate


    @staticmethod
    def decode_data_gcc(RTCPinfo):
        throughput_length = []
        delay_interval_length = []
        throughput = []
        delay_interval = []

        packet_length = int(ord(RTCPinfo[0]) - 48)
        for i in range(0, packet_length):
            throughput_length.append(int(ord(RTCPinfo[i + 1]) - 48))
            delay_interval_length.append(int(ord(RTCPinfo[i + 1 + packet_length]) - 48))
        loss_length = int(ord(RTCPinfo[2 * packet_length + 1]) - 48)
        rtt_length = int(ord(RTCPinfo[2 * packet_length + 2]) - 48)

        pacing_bitrate_length = int(ord(RTCPinfo[2 * packet_length + 3]) - 48)
        target_bitrate_length = int(ord(RTCPinfo[2 * packet_length + 4]) - 48)
        start = 1 + 2 * packet_length + 4

        for i in range(0, packet_length):
            throughput.append(float(RTCPinfo[start: start + throughput_length[i]]) / 2)
            # print("throughput is : ", throughput[i])
            start = start + throughput_length[i]
        for i in range(0, packet_length):
            delay_interval.append(float(RTCPinfo[start: start + delay_interval_length[i]]))
            # print("delay_interval is : ", delay_interval[i])
            start = start + delay_interval_length[i]
        loss = float(RTCPinfo[start: start + loss_length]) / 100
        start = start + loss_length
        rtt = int(RTCPinfo[start: start + rtt_length])

        start = start + rtt_length
        pacing_bitrate = int(RTCPinfo[start:start + pacing_bitrate_length])
        start = start + pacing_bitrate_length
        target_bitrate = int(RTCPinfo[start:start + target_bitrate_length])
        # print('target_bitrate', target_bitrate)
        return packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate, target_bitrate

    # 2020/9/16 use RTCP mean info as input
    def update_online_observation_mean(self, RTCPinfo):
        packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate = self.decode_data(RTCPinfo)
        self.loss_window = loss
        self.delay_interval_window = np.mean(delay_interval)
        self.throughput_window = np.mean(throughput)
        self.delay_window = rtt / 2.0


    # update 4*4
    def update_online_observation(self, RTCPinfo, t):
        print(f"[DEBUG] Updating online observation at time {t}.")
        packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate = self.decode_data(RTCPinfo)
        print(
            f"[DEBUG] Decoded data: packet_length={packet_length}, loss={loss}, rtt={rtt}, throughput={throughput}, pacing_bitrate={pacing_bitrate}"
        )

        # 验证 pacing_bitrate 是否合理
        if pacing_bitrate <= 0:
            print(f"[WARNING] Invalid pacing_bitrate detected: {pacing_bitrate}")

        # 更新 timeseries 和滑动窗口数据
        self.timeseries.append(t)
        self.loss_movewindow.append(loss)
        self.delay_movewindow.append(rtt / 2.0)
        self.throughput_movewindow.append(np.mean(throughput))

        # 确保 timeseries 至少保留 50 个数据点
        if len(self.timeseries) > 50:
            excess_length = len(self.timeseries) - 50
            self.timeseries = self.timeseries[excess_length:]
            self.loss_movewindow = self.loss_movewindow[excess_length:]
            self.delay_movewindow = self.delay_movewindow[excess_length:]
            self.throughput_movewindow = self.throughput_movewindow[excess_length:]

        print(f"[DEBUG] Timeseries length after truncation: {len(self.timeseries)}")
        print(f"[DEBUG] Current timeseries: {self.timeseries}")

        # 更新窗口数据逻辑
        if packet_length >= HISTORY_LEN:
            self.throughput_window = throughput[-HISTORY_LEN:]
            self.delay_interval_window = delay_interval[-HISTORY_LEN:]
            self.loss_window = HISTORY_LEN * [loss]
            self.delay_window = HISTORY_LEN * [rtt / 2.0]
        else:
            self.throughput_window.extend(throughput)
            self.throughput_window = self.throughput_window[-HISTORY_LEN:]
            self.delay_interval_window.extend(delay_interval)
            self.delay_interval_window = self.delay_interval_window[-HISTORY_LEN:]
            self.loss_window.extend([loss] * packet_length)
            self.loss_window = self.loss_window[-HISTORY_LEN:]
            self.delay_window.extend([rtt / 2.0] * packet_length)
            self.delay_window = self.delay_window[-HISTORY_LEN:]

        self.gap_window = HISTORY_LEN * [(pacing_bitrate - self.last_action) / 1e6]
        print(f"[DEBUG] last_action={self.last_action}, pacing_bitrate={pacing_bitrate}")

        self.throughput = np.mean(self.throughput_window)
        self.delay_interval = np.mean(self.delay_interval_window)
        self.loss = np.mean(self.loss_window)
        self.delay = np.mean(self.delay_window)
        print(
            f"[DEBUG] Updated metrics: throughput={self.throughput}, delay_interval={self.delay_interval}, loss={self.loss}, delay={self.delay}"
        )

    # 2020/9/23 start collect all RCTP info during step.size=120
    def update_online_observation_step(self, RTCPinfo):
        packet_length, loss, delay_interval, throughput, rtt, pacing_bitrate, gcc_bitrate = self.decode_data(RTCPinfo)
        self.step_packet_length += packet_length
        if self.step_packet_length <= LEARN_STEP_PACKET and self.step < LEARN_STEP:
            self.throughput_window_step.extend(throughput)
            self.delay_interval_window_step.extend(delay_interval)
            self.loss_window_step.extend(packet_length * [loss])
            self.delay_window_step.extend(packet_length * [rtt / 2.0])
        elif self.step_packet_length > LEARN_STEP_PACKET and self.step <= LEARN_STEP:
            current_len = len(self.loss_window_step)
            if (current_len < LEARN_STEP_PACKET):
                packet_need_input = LEARN_STEP_PACKET - current_len
                self.throughput_window_step.extend(throughput[0:packet_need_input])
                self.delay_interval_window_step.extend(delay_interval[0:packet_need_input])
                self.loss_window_step.extend(packet_need_input * [loss])
                self.delay_window_step.extend(packet_need_input * [rtt / 2.0])
        elif self.step_packet_length <= LEARN_STEP_PACKET and self.step == LEARN_STEP:
            packet_need_push = LEARN_STEP_PACKET - len(self.loss_window_step)
            for i in range(0, packet_need_push):
                self.delay_window_step.append(0)
                self.loss_window_step.append(0)
                self.throughput_window_step.append(0)
                self.delay_interval_window_step.append(0)

    def get_200ms_state(self, t):
        print("[DEBUG] Generating 200ms state.")

        # 动态调整历史数据的截断范围
        max_window_duration = 3  # 最大保留的历史时间范围
        min_window_duration = 2  # 最小截断范围与当前时间差
        if len(self.timeseries) > 0:
            oldest_time = self.timeseries[0]
            dynamic_window_duration = min(max_window_duration, t - oldest_time)
        else:
            dynamic_window_duration = max_window_duration

        # 初始化 start，只保留满足时间范围的数据
        start = 0
        while start < len(self.timeseries) and t - dynamic_window_duration > self.timeseries[start]:
            start += 1

        # 确保截断后的数据仅包含所需的时间范围
        self.timeseries = self.timeseries[start:]
        self.loss_movewindow = self.loss_movewindow[start:]
        self.delay_movewindow = self.delay_movewindow[start:]
        self.throughput_movewindow = self.throughput_movewindow[start:]
        print(f"[DEBUG] Timeseries length after truncation: {len(self.timeseries)}")
        print(f"[DEBUG] Current timeseries: {self.timeseries}")

        loss_200ms = []
        delay_200ms = []
        throughput_200ms = []
        start = 0  # 从截断后的数据开始

        for i in range(10):
            # 计算当前窗口的时间范围
            window_start = t - min_window_duration + 0.2 * i
            window_end = window_start + 0.2

            # 初始化窗口的 start，只选取属于当前窗口的数据
            while start < len(self.timeseries) and self.timeseries[start] < window_start:
                start += 1

            end = start
            while end < len(self.timeseries) and self.timeseries[end] < window_end:
                end += 1

            data_points = end - start
            print(f"[DEBUG] Window {i}: start={start}, end={end}, data_points={data_points}")

            if data_points == 0:
                # 如果没有数据点，使用最近一个窗口的值或设置默认值
                loss_mean = loss_200ms[-1] if loss_200ms else 0
                delay_mean = delay_200ms[-1] if delay_200ms else 0
                throughput_mean = throughput_200ms[-1] if throughput_200ms else 0
            else:
                loss_mean = np.mean(self.loss_movewindow[start:end])
                delay_mean = np.mean(self.delay_movewindow[start:end])
                throughput_mean = np.mean(self.throughput_movewindow[start:end])

            print(
                f"[DEBUG] Window {i}: window_start={window_start:.2f}, window_end={window_end:.2f}, "
                f"start={start}, end={end}, data_points={data_points}, "
                f"loss_mean={loss_mean}, delay_mean={delay_mean}, throughput_mean={throughput_mean}"
            )

            loss_200ms.append(loss_mean)
            delay_200ms.append(delay_mean)
            throughput_200ms.append(throughput_mean)

            start = end  # 更新 start 为下一个窗口的起点

        state = loss_200ms + delay_200ms + throughput_200ms
        print(f"[DEBUG] Computed state: {state}")
        return state

    def clear_step_window(self):
        self.throughput_window_step = []
        self.delay_interval_window_step = []
        self.loss_window_step = []
        self.delay_window_step = []

    def update_concerto_observation(self, RTCPinfo):
        packet_length, loss, delay_interval, throughput, rtt, decode_data = self.decode_data(RTCPinfo)
        if packet_length > HISTORY_LEN:
            packet_length = packet_length - packet_length % HISTORY_LEN
        for i in range(packet_length):
            self.loss_concerto.pop(0)
            self.delay_interval_concerto.pop(0)

            self.loss_concerto.append(loss * 100.0)

        self.delay_interval_concerto.extend(delay_interval[0:packet_length])

        for i in range(packet_length // HISTORY_LEN):
            self.throughput_concerto.pop(0)
            self.throughput_concerto.append(float(np.mean(throughput[i * HISTORY_LEN:(i + 1) * HISTORY_LEN])))

    def as_array_average(self):
        return np.array([self.loss, self.delay, self.delay_interval, self.throughput])

    def as_array_step(self):
        return np.array([np.array(self.loss_window_step),
                         np.array(self.delay_window_step),
                         np.array(self.delay_interval_window_step),
                         np.array(self.throughput_window_step)])

    def as_array(self):
        return np.array([np.array(self.loss_window),
                         np.array(self.delay_window),
                         np.array(self.delay_interval_window),
                         np.array(self.throughput_window)])

    def as_concerto_array(self):
        # print(self.loss_concerto, self.delay_interval_concerto, self.throughput_concerto)
        return np.array([self.loss_concerto + self.delay_interval_concerto + self.throughput_concerto]).reshape(
            [-1, 1, 135])

    def as_array_with_pacing(self):
        return np.array([np.array(self.loss_window),
                         np.array(self.delay_window),
                         np.array(self.delay_interval_window),
                         np.array(self.throughput_window),
                         np.array(self.gap_window)])

    def monitor(self):
        if self.step < self.learning_step:

            self.buffer.append(self.as_array())
            if (self.step == self.learning_step - 1):
                self.buffer = np.array(self.buffer)

    def monitor_clear(self):
        self.buffer = []
