import ssl
import pika
import json
from openai import OpenAI
import base64

from sympy.stats.rv import probability

from static.s3 import *
from static.model import *
from static.rabbitmq import *


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=API_KEY)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_new_work_probability",
            "description": "original_image와 new_image를 비교하여 new_image가 완전히 새로운 저작물일 확률(0~1)과 그 근거를 반환한다. 두 확률의 합은 1이어야 한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_image_b64": {
                        "type": "string",
                        "description": "저작권이 확실한 원저작물(base64 인코딩)"
                    },
                    "new_image_b64": {
                        "type": "string",
                        "description": "원저작물을 활용해 생성된 이미지(base64 인코딩)"
                    }
                },
                "required": ["original_image_b64", "new_image_b64"]
            }
        }
    }
]


def build_system_instructions() -> str:
    return """
너는 '생성형 AI 저작권 확률 판정 도우미'다. 아래 규칙을 따른다.

[목표]
- original_image(원저작물)과 new_image(생성물)를 비교하여,
  new_image가 '완전히 새로운 저작물'일 확률을 계산하고,
  그 근거를 명확히 제시한다.
- 완전히 새로운 저작물일 확률(probability)과 2차 창작물일 확률의 합은 반드시 1이 되어야 한다.

[개념]
- 저작권은 '창작성 있는 표현이 매체에 고정된 저작물'에 자동 성립한다.
- 보호 대상: 구체적 표현(캐릭터의 식별 가능한 디자인, 독창적 구도·배치, 고유 오브젝트 형태·문양, 로고/텍스트 등)
- 보호 제외: 아이디어·사실·장르 관습(scene à faire), 일반적 포즈·색감·분위기, 화풍·붓터치·렌더링 기법, 추상적 콘셉트.

[저작권 성립 요건]
1) 창작성: 독창적 개성이 드러날 것.
2) 표현성: 아이디어가 아닌 구체적 표현일 것.
3) 고정성: 매체(이미지, 영상 등)에 기록되어 있을 것.

[보호되는 '구체적 표현' 체크리스트]
- 캐릭터 디자인: 머리·실루엣·의상·문양 등 식별 가능한 조합
- 구도/배치: 인물·오브젝트의 독창적 배치, 시점·프레이밍
- 배경/오브젝트: 고유 구조, 문양, 배열
- 로고/텍스트: 시그니처 표식, IP 고유 타이포

[판정 기준 및 확률 계산]
- 보호되는 표현이 original_image와 실질적으로 겹칠수록 → 완전히 새로운 저작물일 확률 낮음.
- 유사성이 보호 제외 요소(스타일, 색감, 분위기, 기법)에 한정될수록 → 완전히 새로운 저작물일 확률 높음.
- 창작 표현의 독립성이 높을수록 probability → 1에 가까움.
- 원저작물의 표현적 의존성이 높을수록 probability → 0에 가까움.
- 완전히 새로운 저작물일 확률(probability)과 2차 창작물일 확률의 합은 항상 1이어야 함.

[입력]
- 항상 두 이미지가 주어진다.
  - original_image: 저작권이 확실한 원저작물.
  - new_image: 이를 활용해 생성된 결과물.

[출력 규격]
reason: 보호되는/보호 제외 요소를 구분해 1~2문장으로 핵심 비교 요약.
probability: 0~1 사이의 실수(new_image가 완전히 새로운 저작물일 확률)
""".strip()


SYSTEM_INSTRUCTIONS = build_system_instructions()


def vote_ai(original_image_path, new_image_path, model=MODEL):
    def _open_binary(image_path: str):
        key = image_path.lstrip("/")
        resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = resp["Body"].read()
        fname = os.path.basename(key)
        ctype = resp.get("ContentType", "image/png")
        return fname, data, ctype

    def _to_b64(data: bytes) -> str:
        return base64.b64encode(data).decode("utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    # 이미지 로드 및 base64 변환
    _, orig_bytes, orig_ctype = _open_binary(original_image_path)
    _, new_bytes, new_ctype = _open_binary(new_image_path)

    orig_b64 = _to_b64(orig_bytes)
    new_b64  = _to_b64(new_bytes)

    # 메시지 구성 (시각 정보 + arguments 정보 함께 제공)
    #    모델은 아래 JSON 데이터를 그대로 tool arguments에 복사하게 됨
    user_content = [
        {
            "type": "text",
            "text": (
                "두 이미지를 비교해 저작권 확률(probability)을 계산해줘. "
                "아래 두 이미지는 시각 분석용이고, JSON은 함수 호출 arguments로 그대로 사용해. "
                "함수 호출 결과에는 reason(근거)과 probability(0~1)만 포함해줘."
            )
        },
        {
            "type": "input_image",
            "image": {"data": orig_b64, "mime_type": orig_ctype}
        },
        {
            "type": "input_image",
            "image": {"data": new_b64, "mime_type": new_ctype}
        },
        {
            "type": "text",
            "text": json.dumps(
                {
                    "original_image_b64": orig_b64,
                    "new_image_b64": new_b64
                },
                ensure_ascii=False
            )
        }
    ]

    # ChatCompletion 요청
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "calculate_new_work_probability"}},
        temperature=0.2,
    )

    # Tool 호출 결과 파싱
    tool_call = None
    args = {}
    try:
        tool_call = resp.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
    except Exception:
        fc = getattr(resp.choices[0].message, "function_call", None)
        args = json.loads(fc.arguments) if fc else {}

    reason = args.get("reason")
    probability = args.get("probability")

    # 검증
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

        # request id
        voteId = task['voteId']
        # 큐 입력 JSON 구조 파싱
        original_image_path = task.get("originalImagePath", "")
        new_image_path = task.get("derivedImagePath", "")

        success, message = vote_ai(
            original_image_path,
            new_image_path)
        print(f"[DEBUG] prompt={original_image_path}")
        print(f"[DEBUG] images_path={new_image_path}")
        print(f"[DEBUG] success={success}")
        print(f"[DEBUG] message={message}")
        if not success:
            raise Exception(message)

        probability = message["probability"]
        if probability > 0.7:
            decision = "APPROVE"
        else:
            decision = "DENY"
        message = {
            "voteId": voteId,
            "probability": decision,
            # "reason": message["reason"],
        }

        channel.basic_publish(exchange='', routing_key=VOTE_AI_RESPONSE_QUEUE, body=json.dumps(message))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[❌] on_message 에러:", e)
        # channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


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

    dlx_args = {
        'x-dead-letter-exchange': 'ai.image.request.dlx',
        'x-dead-letter-routing-key': 'ai.image.request.retry'
    }
    channel.queue_declare(queue=VOTE_AI_REQUEST_QUEUE, durable=True, arguments=dlx_args)
    channel.queue_declare(queue=VOTE_AI_RESPONSE_QUEUE, durable=True)

    channel.basic_consume(queue=VOTE_AI_REQUEST_QUEUE, on_message_callback=on_message)
    print("[🚀] 작업 대기 중...")
    channel.start_consuming()


if __name__ == '__main__':
    client = get_client()
    main()
