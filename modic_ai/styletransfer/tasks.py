from .model_loader import get_model
from .style_transfer import run_style_transfer
import subprocess
import time


def wait_for_result(content, style, prompt, preprocessor):
    try:
        # while not get_gpu_memory():
        #     time.sleep(10)
        model_ref = get_model(preprocessor)
        result = run_style_transfer(content, style, prompt, model_ref)

        return result

    except Exception as e:
        print(e)
        return None

def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']
    )
    used, total = map(int, result.decode().strip().split(','))
    if total - used > 5000:
        return True
    else:
        return False
