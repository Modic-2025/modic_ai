import os
import ssl
import json
import base64
import io
import pika
from openai import OpenAI

from static.s3 import *
from static.model import *
from static.rabbitmq import *


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


# 함수 툴: 결과만 받도록 간소화
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_new_work_probability",
            "description": "두 이미지를 비교해 new_image가 완전히 새로운 저작물일 확률(0~1)과 그 근거(reason)를 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["reason", "probability"]
            }
        }
    }
]


def build_system_instructions() -> str:
    return """
너는 '생성형 AI 저작권 확률 판정 도우미'다.
두 이미지를 비교해 보호되는 표현의 중복 여부를 근거로 reason(1~2문장)과 probability(0~1)를 산출하고,
함수 calculate_new_work_probability를 호출하여 그 값을 반환한다.
""".strip()


SYSTEM_INSTRUCTIONS = build_system_instructions()


def _open_binary(image_path: str):
    key = image_path.lstrip("/")
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    data = resp["Body"].read()
    ctype = resp.get("ContentType", "image/png")
    return data, ctype


def _shrink_image_jpeg(data: bytes, max_side: int = 1024, quality: int = 85) -> bytes:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            new_size = (int(w / scale), int(h / scale))
            im = im.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _data_url(b64: str, mime: str) -> str:
    # Chat Completions는 image_url만 지원 → data URL로 전달
    return f"data:{mime};base64,{b64}"


def _build_user_content(orig_b64: str, orig_ctype: str, new_b64: str, new_ctype: str):
    return [
        {
            "type": "text",
            "text": (
                "아래 두 이미지를 비교해 주세요. "
                "보호되는 표현(캐릭터 식별요소/독창적 구도/고유 문양·로고 등)을 기준으로 reason을 1~2문장으로 쓰고, "
                "probability(0~1)를 산출한 뒤 함수(calculate_new_work_probability)를 호출하세요."
            )
        },
        {   # 👇 Chat Completions 형식: image_url + data URL
            "type": "image_url",
            "image_url": {"url": _data_url(orig_b64, orig_ctype)}
        },
        {
            "type": "image_url",
            "image_url": {"url": _data_url(new_b64, new_ctype)}
        },
    ]


def _call_model(user_content, model):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",   "content": user_content},
        ],
        tools=TOOLS,
        temperature=0.2,
    )


def vote_ai(original_image_path, new_image_path, model=MODEL):
    # 1) 원본 로드
    orig_bytes, orig_ctype = _open_binary(original_image_path)
    new_bytes,  new_ctype  = _open_binary(new_image_path)

    # 2) 1차 시도
    try:
        user_content = _build_user_content(_b64(orig_bytes), orig_ctype, _b64(new_bytes), new_ctype)
        resp = _call_model(user_content, model)
    except Exception as e:
        # 예외 발생 시 바로 축소 재시도 분기로
        resp = None
        last_err = e

    # 3) 429 또는 요청 크기 문제일 경우 → 축소 후 1회 재시도
    def parse_tool_args(response):
        args = {}
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
        except Exception:
            fc = getattr(response.choices[0].message, "function_call", None)
            if fc:
                try:
                    args = json.loads(fc.arguments)
                except Exception:
                    pass
        return args

    args = parse_tool_args(resp) if resp else {}
    if not args:
        # 축소본으로 재시도
        s_orig = _shrink_image_jpeg(orig_bytes, max_side=1024, quality=80)
        s_new  = _shrink_image_jpeg(new_bytes,  max_side=1024, quality=80)
        user_small = _build_user_content(_b64(s_orig), orig_ctype, _b64(s_new), new_ctype)
        resp = _call_model(user_small, model)
        args = parse_tool_args(resp)

    reason = args.get("reason")
    probability = args.get("probability")

    if not isinstance(reason, str) or not reason.strip():
        return False, "reason is None."
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return False, "probability is invalid."
    if not (0.0 <= probability <= 1.0):
        return False, "probability out of range."
    return True, {"reason": reason.strip(), "probability": probability}


def on_message(channel, method, properties, body):
    try:
        print("[📥] 작업 수신:", body.decode("utf-8"))
        task = json.loads(body)

        voteId = task['voteId']
        original_image_path = task.get("originalImagePath", "")
        new_image_path = task.get("derivedImagePath", "")

        success, payload = vote_ai(original_image_path, new_image_path)
        print(f"[DEBUG] success={success}")
        print(f"[DEBUG] payload={payload}")

        if not success:
            raise RuntimeError(str(payload))

        prob = float(payload["probability"])
        decision = "APPROVE" if prob > 0.7 else "DENY"

        result = {"voteId": voteId, "probability": prob, "decision": decision}
        channel.basic_publish(exchange='', routing_key=VOTE_AI_RESPONSE_QUEUE, body=json.dumps(result))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[❌] on_message 에러:", e)
        # DLX/FAILED_QUEUE 없이 그냥 ack해서 루프 방지
        channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    context = ssl.create_default_context()
    credentials = pika.PlainCredentials(VOTE_AI_USERNAME, VOTE_AI_PASSWORD)
    params = pika.ConnectionParameters(
        host=VOTE_AI_HOST,
        port=int(VOTE_AI_PORT),
        credentials=credentials,
        ssl_options=pika.SSLOptions(context)
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=VOTE_AI_REQUEST_QUEUE, durable=True)
    channel.queue_declare(queue=VOTE_AI_RESPONSE_QUEUE, durable=True)

    channel.basic_consume(queue=VOTE_AI_REQUEST_QUEUE, on_message_callback=on_message)
    print("[🚀] 작업 대기 중...")
    channel.start_consuming()


if __name__ == '__main__':
    client = get_client()
    main()
