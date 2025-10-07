import subprocess
from ..static.minimum_gpu_memory import STYTR2_MIN_GPU_MEM

def get_gpu_memory(model_name):
    if model_name == "StyTR2":
        min_gpu_mem = STYTR2_MIN_GPU_MEM
    else:
        print(f"There's no {model_name} in the list.")
        return False
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']
    )
    used, total = map(int, result.decode().strip().split(','))
    if total - used > min_gpu_mem:
        return True
    else:
        return False
