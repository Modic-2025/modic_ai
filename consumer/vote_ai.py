import os
from io import BytesIO
import ssl
import pika
import json
from openai import OpenAI

from static.s3 import *
from static.model import *


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=API_KEY)


def vote_ai(original_image_path, new_image_path, api_key):
    def _open_binary(image_path: str):
        key = image_path.lstrip("/")
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)  # <-- Key=key 로!
        data = resp["Body"].read()
        fname = os.path.basename(key)
        ctype = resp.get("ContentType", "image/png")  # S3에 저장한 ContentType 재사용
        return fname, BytesIO(data), ctype

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    images = {}
    original_name, original_fh, original_ct = images_open_binary(original_image_path)
    new_name, new_fh, new_ct = _open_binary(original_image_path)

def on_message(channel, method, properties, body):
    try:
        print("[📥] 작업 수신:", body.decode("utf-8"))
        task = json.loads(body)

        # request id
        request_id = task['requestId']
        # 큐 입력 JSON 구조 파싱
        original_image_path = task.get("originalImagePath", "")
        new_image_path = task.get("newImagePath", "")

        message = {
            "isSuccess": True,
            "requestId": request_id,
            "isImageGenerated": True,
            "imagePath": message["image_path"],
            "fullImageName": message["file_name"],
            "imageName": message["image_name"],
            "extension": "PNG",
            "description": message["description"],
            "chatSummary": message["chat_summary"],
            "fromStyleImage": message["style_transfer"]
        }

        channel.basic_publish(exchange='', routing_key=RESPONSE_QUEUE, body=json.dumps(message))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[❌] on_message 에러:", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    context = ssl.create_default_context()
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=int(RABBITMQ_PORT),
        credentials=credentials,
        ssl_options=pika.SSLOptions(context)
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    dlx_args = {
        'x-dead-letter-exchange': 'ai.image.request.dlx',
        'x-dead-letter-routing-key': 'ai.image.request.retry'
    }
    channel.queue_declare(queue=REQUEST_QUEUE, durable=True, arguments=dlx_args)
    channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)

    channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=on_message)
    print("[🚀] 작업 대기 중...")
    channel.start_consuming()


if __name__ == '__main__':
    client = get_client()
    main()