from .StyleShot.style_shot import StyleShotModel
import time

from celery import shared_task
from django.db import transaction
import uuid

from .StyTR2.stytr2 import StyTR2
from .utills import get_gpu_memory


def wait_for_result(content, style, prompt, preprocessor):
    try:
        while not get_gpu_memory("StyTR2"):
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
