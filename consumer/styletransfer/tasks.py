import time

from .StyTR2.stytr2 import StyTR2
from .utills import get_gpu_memory


def wait_for_result(content, style, prompt, preprocessor):
    try:
        while not get_gpu_memory("StyTR2"):
            time.sleep(10)

        strtr2 = StyTR2()
        result = strtr2.inference(content, style)

        return result

    except Exception as e:
        print(e)
        return None
