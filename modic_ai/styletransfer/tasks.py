import subprocess
from .StyleShot.style_shot import StyleShotModel
from .StyTR2.stytr2 import StyTR2

import time

# from celery import shared_task


def wait_for_result(content, style, prompt, preprocessor):
    try:
        while not get_gpu_memory():
            time.sleep(10)
        # model_ref = get_model(preprocessor)

        # style_shot = StyleShotModel()
        # result = style_shot.inference(content, style, prompt, preprocessor)
        strtr2 = StyTR2()
        result = strtr2.inference(content, style)

        return result

    except Exception as e:
        print(e)
        return None

def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']
    )
    used, total = map(int, result.decode().strip().split(','))
    if total - used > 1000:
        return True
    else:
        return False


# @shared_task
# def run_style_transfer(content, style, prompt, preprocessor):
#     model_ref = get_model(preprocessor)
#     result = run_style_transfer(content, style, prompt, model_ref)
#
#     return result
