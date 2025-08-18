import os
import multiprocessing as mp
from multiprocessing import Manager
import subprocess
import time
from typing import List, Tuple
import threading

# GPU管理的全局信号量
gpu_semaphores = {i: threading.Semaphore(1) for i in range(8)}  # 假设有8张GPU
gpu_lock = threading.Lock()

# 全局变量：每个GPU的任务上限和当前任务数
gpu_max_tasks = 2  # 每个GPU最多同时运行的任务数

# 使用Manager创建共享变量
manager = Manager()
gpu_task_counts = manager.dict({i: 0 for i in range(8)})  # 共享任务计数
gpu_lock = manager.Lock()  # 共享锁

def get_gpu_memory_usage() -> List[Tuple[int, int]]:
    """
    获取所有GPU的显存使用情况
    返回: [(gpu_id, free_memory), ...]
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', 
                                       '--format=csv,nounits,noheader'])
        lines = output.decode().strip().split('\n')
        gpu_memory = []
        for line in lines:
            gpu_id, free_memory = map(int, line.split(','))
            gpu_memory.append((gpu_id, free_memory))
        return gpu_memory
    except:
        return []

def get_free_gpu(min_memory: int = 1000) -> int:
    """
    获取显存空闲且任务数未达上限的GPU，如果没有满足条件的GPU则返回None
    :param min_memory: 最小需要的显存(MB)
    :return: GPU ID或None
    """
    with gpu_lock:
        gpu_memory = get_gpu_memory_usage()
        if not gpu_memory:
            return None
        
        # 按照可用显存排序
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        
        # 检查每个GPU的显存、信号量状态和任务数
        for gpu_id, free_memory in gpu_memory:
            if (
                free_memory >= min_memory
                and gpu_semaphores[gpu_id]._value > 0
                and gpu_task_counts[gpu_id] < gpu_max_tasks
            ):
                return gpu_id
        return None

def wait_for_free_gpu(min_memory: int = 1000, check_interval: int = 10) -> int:
    """
    等待直到有空闲的GPU（显存足够且任务数未达上限）
    :param min_memory: 最小需要的显存(MB)
    :param check_interval: 检查间隔(秒)
    :return: GPU ID
    """
    while True:
        gpu_id = get_free_gpu(min_memory)
        if gpu_id is not None:
            return gpu_id
        print(f"No GPU available with {min_memory}MB free memory and task count < {gpu_max_tasks}, waiting {check_interval} seconds...")
        time.sleep(check_interval)

def worker(input_file: str, output_dir: str, *args, **kwargs):
    """
    工作进程函数，处理单个文件
    """
    try:
        gpu_id = wait_for_free_gpu()
        
        with gpu_lock:
            if gpu_task_counts[gpu_id] >= gpu_max_tasks:
                print(f"GPU {gpu_id} has reached max tasks ({gpu_max_tasks}), retrying...")
                return worker(input_file, output_dir, *args, **kwargs)
            
            gpu_task_counts[gpu_id] += 1
            gpu_semaphores[gpu_id].acquire()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Processing {input_file} on GPU {gpu_id} (Tasks: {gpu_task_counts[gpu_id]}/{gpu_max_tasks})")
        
        run_tts_with_list_file(input_file, output_dir, *args, **kwargs)
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
    finally:
        if 'gpu_id' in locals():
            with gpu_lock:
                gpu_task_counts[gpu_id] -= 1
                gpu_semaphores[gpu_id].release()

def process_all_list_files(func, input_root, output_root, extension='.list', num_workers=8, *args, **kwargs):
    """
    并行处理所有文件
    
    :param func: 处理函数
    :param input_root: 输入文件根目录
    :param output_root: 输出文件根目录
    :param extension: 文件扩展名
    :param num_workers: 并行进程数，默认为8
    """
    # 收集所有需要处理的文件
    tasks = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(extension):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                
                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)
                tasks.append((input_file, output_dir))

    print(f"Found {len(tasks)} files to process")
    
    # 创建进程池
    with mp.Pool(num_workers) as pool:
        # 使用starmap并行处理任务
        pool.starmap(worker, [(f, d) + args for f, d in tasks], kwargs)

def run_tts_with_list_file(input_file, output_dir, *args, **kwargs):
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                print(line)
                # 这里添加你的TTS处理逻辑

def main():
    process_all_list_files(run_tts_with_list_file, 
                          '/data/tts/data/list', 
                          '/data/tts/data/infer',
                          num_workers=8)

if __name__ == '__main__':
    main()
