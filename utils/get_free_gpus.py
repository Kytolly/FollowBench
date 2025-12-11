import pynvml
import sys

def get_free_gpu_ids(min_memory_free=96000, max_utilization=10):
    """
    检测空闲显卡并返回以逗号分隔的 ID 字符串。
    :param min_memory_free: 最小剩余显存 (MB)，默认 90GB
    :param max_utilization: 最大 GPU 利用率 (%)
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_ids = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        free_mem_mb = info.free / 1024 / 1024
        gpu_util = util.gpu
        if free_mem_mb > min_memory_free and gpu_util < max_utilization:
            free_ids.append(str(i))

    pynvml.nvmlShutdown()
    
    if not free_ids:
        return ""   
    return ",".join(free_ids)