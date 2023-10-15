import os
from configs import *

def exec_benchmark(recommend_command, opt):
    os.system(f'sshpass -p {BENCHMARK_PW} ssh {BENCHMARK_ID}@{BENCHMARK_IP} "{recommend_command}"')
    if os.path.exists('res.txt'):
        os.system('rm res.txt')
    os.system(f'sshpass -p {BENCHMARK_PW} ssh {BENCHMARK_ID}@{BENCHMARK_IP} "python3 show_ex.py" >> res.txt')
