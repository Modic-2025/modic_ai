import django
import os
import asyncio
from styletransfer.model_loader import preload_models
from styletransfer.worker import worker

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_backend.settings')
django.setup()

# 모델 GPU 로딩 및 워커 시작
preload_models()
asyncio.get_event_loop().create_task(worker())

from django.core.asgi import get_asgi_application
application = get_asgi_application()
