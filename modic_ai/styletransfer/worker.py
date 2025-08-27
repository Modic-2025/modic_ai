import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import pika
import ssl
import json
import uuid
import requests
import boto3
from botocore.client import Config

from io import BytesIO

from .tasks import wait_for_result

# S3 setup
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET = os.getenv('S3_BUCKET')
PUBLIC_BASE_URL = os.getenv('PUBLIC_BASE_URL')
S3_PATH_PREFIX = os.getenv('S3_PATH_PREFIX', '/generated-images')

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    config=Config(signature_version='s3v4')
)

# RabbitMQ
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
REQUEST_QUEUE = os.getenv('REQUEST_QUEUE')
RESPONSE_QUEUE = os.getenv('RESPONSE_QUEUE')
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT')
RABBITMQ_USERNAME = os.getenv('RABBITMQ_USERNAME')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')

def get_s3_key():
    image_name = uuid.uuid4().hex
    extension = "png"
    full_image_name = f"{image_name}.{extension}"
    prefix = S3_PATH_PREFIX.strip('/')
    s3_key = f"{prefix}/{full_image_name}"
    return s3_key, full_image_name, image_name, extension.upper()

def download_image(url):
    res = requests.get(url)
    res.raise_for_status()
    return res.content

def upload_to_s3(image_bytes, s3_key):
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType='image/png'
    )

# 🔹 새로운 스펙에 맞춘 결과 메시지 전송
def send_result(channel, request_id, s3_key, full_image_name, image_name, extension):
    image_url = f"{PUBLIC_BASE_URL}/{s3_key}"
    image_path = f"/{s3_key}"
    message = {
        "requestId": request_id,
        "imageUrl": image_url,
        "imagePath": image_path,
        "fullImageName": full_image_name,
        "imageName": image_name,
        "extension": extension
    }
    channel.basic_publish(exchange='', routing_key=RESPONSE_QUEUE, body=json.dumps(message))
    print(f"[✅] 결과 전송 완료: {message}")

def on_message(channel, method, properties, body):
    try:
        print("[📥] 작업 수신:", body)
        task = json.loads(body)
        request_id = task['requestId']
        content_image_url = task['requestImageUrl']
        style_image_urls = task['styleImageUrls']

        content_image = download_image(content_image_url)
        style_images = [download_image(url) for url in style_image_urls]

        content_image = BytesIO(content_image)
        style_image = BytesIO(style_images[0])

        result_image = wait_for_result(content_image, style_image, prompt=None, preprocessor=None)

        s3_key, full_image_name, image_name, extension = get_s3_key()
        upload_to_s3(result_image, s3_key)

        send_result(channel, request_id, s3_key, full_image_name, image_name, extension)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[❌] 에러 발생:", e)
        channel.basic_nack(delivery_tag=method.delivery_tag)

def main():
    context = ssl.create_default_context()
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        ssl_options=pika.SSLOptions(context)
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)
    channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=on_message)
    print("[🚀] 작업 대기 중...")
    channel.start_consuming()

if __name__ == '__main__':
    main()
