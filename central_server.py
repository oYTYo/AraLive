import os
import signal
import socketserver
import socket
from gym.envs.registration import register
from stable_baselines.common.policies import register_policy
from multiprocessing import Process, Pipe
from single_trainer_newppo import Trainer
from utils import current_time, rl_log
import warnings
import time

# 忽略警告信息
warnings.filterwarnings('ignore')

HOST = ""
PORT = 9999
IS_TEST = False

register(id='OnlineEnv-v0', entry_point='single_trainer_newppo:OnlineEnvironment')

def train(pipe, directory):
    f = open('config.txt', mode='r', encoding='utf-8')
    params = [150, 5, 80, 12]
    i = 0
    for line in f:
        v = int(line.strip('\n'))
        if i < 4 and v > 0:
            params[i] = v
            i += 1
        else:
            break
    f.close()
    rl_log(flag='INFO', content=str(params))
    t = Trainer(pipe, directory)
    t.set_time_steps(total_steps=1024)
    t.set_reward_parameters(
        throughput_param=params[0],
        delay_param=params[1],
        loss_param=params[2],
        smooth_param=params[3]
    )
    t.run_actor()

class CentralServer(socketserver.BaseRequestHandler):
    def handle(self):
        ip, port = self.client_address
        self.request.setblocking(0)
        self.request.settimeout(30)

        try:
            info = self.request.recv(1024)
            info = info.decode('utf-8')
            if info.find('%%%%%') == -1:
                rl_log(flag="ERROR", content="Abnormal Request.")
                return
        except UnicodeDecodeError as e:
            rl_log(flag="ERROR", content="Data Exception.")
            return

        id_index = info.find('ID[')
        isp_index = info.find('ISP[')
        net_index = info.find('NET[')
        time_index = info.find('TIME[')
        end_index = info.find(']%')

        user_id = 'default'
        isp = 'default'
        net = 'default'
        start_time = 'default'

        if id_index != -1 and isp_index != -1:
            user_id = info[id_index + 3:isp_index - 1]

        if isp_index != -1 and net_index != -1:
            isp = info[isp_index + 4:net_index - 1]

        if net_index != -1 and time_index != -1:
            net = info[net_index + 4:time_index - 1]

        if time_index != -1 and end_index != -1:
            start_time = info[time_index + 5:end_index]

        user_mark = user_id + '_' + isp + '_' + net + '_' + start_time

        if user_id == 'default':
            return

        tcp_end, trainer_end = Pipe(True)

        if IS_TEST:
            sub_process = Process(target=test, name='tester', args=(trainer_end,))
        else:
            sub_process = Process(target=train, name='trainer', args=(trainer_end, str(user_mark)))

        sub_process.start()

        data_stream = ""
        while True:
            try:
                try:
                    data = self.request.recv(1024)
                    recv_time = time.time()
                    data = data.decode('utf-8')
                    rl_log(flag="DEBUG", content=f"Received data at {recv_time:.6f}: {data[:50]}")  # Log received data time
                except UnicodeDecodeError as e:
                    rl_log(flag="ERROR", content="Data Exception.")
                    send_data = "1000000"
                    send_data += (1024 - len(send_data)) * '%'
                    self.request.sendall(send_data.encode('utf-8'))
                    continue

                data_stream += str(data)
                if data_stream.find('[') != -1:
                    data_stream = data_stream[data_stream.find('['):]

                if len(data_stream) < 1024:
                    continue
                else:
                    begin_index = data_stream.find('[')
                    end_index = data_stream.find(']')
                    if begin_index != -1 and end_index != -1:
                        real_data = data_stream[begin_index + 1:end_index]
                        real_data += (1024 - len(real_data)) * '%'
                        tcp_end.send_bytes(real_data.encode('utf-8'), 0, 1024)
                        rl_log(flag="DEBUG", content=f"Sent data to trainer at {time.time():.6f}.")  # Log send data time

                        data_stream = data_stream[end_index + 1:]
                        action = tcp_end.recv_bytes(1024)
                        self.request.sendall(action)
                        rl_log(flag="DEBUG", content=f"Sent action back to client at {time.time():.6f}.")  # Log action send time

            except (ConnectionError, socket.timeout) as e:
                rl_log(flag="INFO", user_id=str(user_id), content="Connection Closed.")
                stop_signal = "STOP" + 1020 * '*'
                tcp_end.send_bytes(stop_signal.encode('utf-8'), 0, 1024)
                command = tcp_end.recv()
                if command == 'KILL':
                    os.kill(sub_process.pid, signal.SIGTERM)
                    rl_log(flag="INFO", user_id=str(user_id), content=str(sub_process.pid) + " was killed.")
                    tcp_end.close()
                    self.request.close()
                return

if __name__ == '__main__':
    s = socketserver.ThreadingTCPServer((HOST, PORT), CentralServer)
    s.serve_forever()
