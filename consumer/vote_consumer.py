import os
import ssl
import io
import json
import base64
import random
from typing import List, Dict, Tuple
import pika
from openai import OpenAI

from static.s3 import *
from static.model import *
from static.rabbitmq import *


# ============================== 공통 유틸 ==============================
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def _open_binary(image_path: str):
    key = image_path.lstrip("/")
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    data = resp["Body"].read()
    ctype = resp.get("ContentType", "image/png")
    return io.BytesIO(data), ctype


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _data_url(b64: str, mime: str) -> str:
    return f"data:{mime};base64,{b64}"


# ============================== 1. Abstraction Agent ==============================
SYSTEM_INSTRUCTIONS_ABS = """
너는 'Abstraction Agent'다.
두 이미지를 비교해 각각의 추상적 표현(구성요소, 테마, 색상, 구도 등)을 텍스트로 분석한다.
결과는 다음 형식을 따라야 한다:

Image1: [요약된 구성 요소 묘사]
Image2: [요약된 구성 요소 묘사]
""".strip()


def _build_user_content_abstraction(orig_b64: str, orig_ctype: str, new_b64: str, new_ctype: str):
    return [
        {
            "type": "text",
            "text": (
                "두 이미지를 추상화하여 각각의 구성 요소를 설명해 주세요. "
                "각 이미지를 다음 형식으로 분석하세요:\n"
                "Image1: [요약된 구성 요소]\n"
                "Image2: [요약된 구성 요소]\n\n"
                "구성 요소에는 구도(composition), 테마(themes), 색상(color palette), 시각 요소(visual elements)를 포함하세요."
            )
        },
        {"type": "image_url", "image_url": {"url": _data_url(orig_b64, orig_ctype)}},
        {"type": "image_url", "image_url": {"url": _data_url(new_b64, new_ctype)}},
    ]


def run_abstraction_agent(original_image_path, new_image_path, model=MODEL):
    try:
        orig_fh, orig_ctype = _open_binary(original_image_path)
        new_fh, new_ctype = _open_binary(new_image_path)
        orig_b64 = _b64(orig_fh.read())
        new_b64 = _b64(new_fh.read())

        user_content = _build_user_content_abstraction(orig_b64, orig_ctype, new_b64, new_ctype)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS_ABS},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        message = resp.choices[0].message.content.strip()
        result = {}
        for line in message.splitlines():
            if line.startswith("Image1:"):
                result["abstract1"] = line.split(":", 1)[1].strip()
            elif line.startswith("Image2:"):
                result["abstract2"] = line.split(":", 1)[1].strip()

        if "abstract1" not in result or "abstract2" not in result:
            return False, f"abstraction 결과 파싱 실패: {message}"
        return True, result
    except Exception as e:
        return False, str(e)


# ============================== 2. Filtering Agent ==============================
SYSTEM_INSTRUCTIONS_FIL = """
너는 'Filtering Agent'다.
입력으로 받은 두 이미지의 구성 요소 설명에서 비보호(unprotectable) 표현(공통된 테마/공공 영역/기능적 구성요소 등)을 제거하고,
창작성이 있는 저작권 보호 요소만 골라 다음 형식으로 출력한다:

Image1 Unique Elements: [보호 대상 표현 목록 또는 요약]
Image2 Unique Elements: [보호 대상 표현 목록 또는 요약]
""".strip()


def _build_user_content_filtering(abstract1: str, abstract2: str):
    return [
        {
            "type": "text",
            "text": (
                f"다음은 두 이미지의 추상화된 구성 요소 설명입니다. "
                "각 설명에서 저작권 보호 불가 표현(공통 테마, 기능성, 공공 도메인 등)을 제거하고, "
                "남은 고유하고 창작적인 표현 요소만 추려서 아래 형식으로 출력하세요:\n\n"
                f"Image1: {abstract1}\n"
                f"Image2: {abstract2}\n\n"
                "결과 형식:\n"
                "Image1 Unique Elements: ...\n"
                "Image2 Unique Elements: ..."
            )
        }
    ]


def run_filtering_agent(abstract1: str, abstract2: str, model=MODEL):
    try:
        user_content = _build_user_content_filtering(abstract1, abstract2)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS_FIL},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        message = resp.choices[0].message.content.strip()
        result = {}
        for line in message.splitlines():
            if line.startswith("Image1 Unique Elements:"):
                result["image1"] = line.split(":", 1)[1].strip()
            elif line.startswith("Image2 Unique Elements:"):
                result["image2"] = line.split(":", 1)[1].strip()
        if "image1" not in result or "image2" not in result:
            return False, f"filtering 결과 파싱 실패: {message}"
        return True, result
    except Exception as e:
        return False, str(e)


# ============================== 3. Two-Sided Debate Agent ==============================
_KR_COPYRIGHT_GUIDE = """
[한국 저작권법 인지용 요약 가이드]
- 아이디어는 비보호, 창작적 표현만 보호.
- 판단 흐름 권장 단계:
  1) 보호대상 식별
  2) 접근가능성/유사성
  3) 변형·창작성
  4) 대체효과/시장영향
  5) 결론: derivative / uncertain / new
"""

_DEFENSE_NEW_SYSTEM = _KR_COPYRIGHT_GUIDE + """
[당신의 역할]
- 입장: "새로운 창작물" 측 대리인(Defense-New)
- 전략: A의 독창적 표현, 변형, 차이점을 강조
- 출력: 120~180자 한국어 문단
"""

_PROSECUTION_DERIV_SYSTEM = _KR_COPYRIGHT_GUIDE + """
[당신의 역할]
- 입장: "기존 창작물" 측 대리인(Prosecution-Deriv)
- 전략: 보호대상 표현의 중복을 근거로 파생 주장
- 출력: 120~180자 한국어 문단
"""

_ARBITER_SYSTEM = _KR_COPYRIGHT_GUIDE + """
[당신은 중립 심판(Arbiter)이다]
- 두 입장의 주장을 종합하여 JSON으로 결과를 내라.
{
  "similarity": 0-100,
  "transformative_degree": 0-100,
  "market_substitution_risk": 0-100,
  "verdict": "derivative|uncertain|new",
  "reasons": ["핵심 근거들"],
  "risk_notes": ["리스크 요약"]
}
"""


def _s3_to_data_url_for_debate(path: str) -> Tuple[str, str]:
    fh, ctype = _open_binary(path)
    raw = fh.read()
    mime = ctype or "image/png"
    return mime, f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _img_block(label: str, url: str):
    return [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": url}}]


def _call_text(model: str, system: str, user: List[dict], hist: List[dict], temperature=0.2):
    msgs = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": user}]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


def _call_json(model: str, system: str, user: List[dict]):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.15)
    txt = (resp.choices[0].message.content or "").strip()
    return json.loads(txt)


def run_two_sided_debate(base_path: str, ref_paths: List[str], rounds: int = 3, model: str = MODEL):
    _, base_url = _s3_to_data_url_for_debate(base_path)
    ref_urls = [_s3_to_data_url_for_debate(p)[1] for p in ref_paths]
    ctx_blocks = _img_block("A(심사 대상)", base_url)
    for i, u in enumerate(ref_urls, 1):
        ctx_blocks += _img_block(f"B#{i}(원본 후보)", u)

    defense_hist, pros_hist, debate = [], [], []
    for r in range(1, rounds + 1):
        defense_user = ctx_blocks + [{"type": "text", "text": f"[라운드 {r}] 새로운 창작물 측 주장"}]
        d_text = _call_text(model, _DEFENSE_NEW_SYSTEM, defense_user, defense_hist)
        defense_hist.append({"role": "assistant", "content": [{"type": "text", "text": d_text}]})

        pros_user = ctx_blocks + [{"type": "text", "text": f"[라운드 {r}] 기존 창작물 측 반박"}]
        p_text = _call_text(model, _PROSECUTION_DERIV_SYSTEM, pros_user, pros_hist)
        pros_hist.append({"role": "assistant", "content": [{"type": "text", "text": p_text}]} )
        debate.append({"round": r, "defense_new": d_text, "prosecution_deriv": p_text})

    arbiter_user = ctx_blocks + [
        {"type": "text", "text": "다음은 토론 로그(JSON)입니다."},
        {"type": "text", "text": json.dumps(debate, ensure_ascii=False)}
    ]
    arbiter = _call_json(model, _ARBITER_SYSTEM, arbiter_user)
    sim = int(arbiter.get("similarity", 0))
    trans = int(arbiter.get("transformative_degree", 0))
    subrisk = int(arbiter.get("market_substitution_risk", 0))
    verdict = arbiter.get("verdict", "uncertain")
    new_prob = max(0.0, min(1.0, (trans / 100) * (1 - sim / 100) * (1 - subrisk / 100)))
    summary = {
        "avg_similarity": sim,
        "transformative_degree": trans,
        "market_substitution_risk": subrisk,
        "verdict": verdict,
        "new_work_probability": round(new_prob, 4),
        "reasons": arbiter.get("reasons", []),
        "risk_notes": arbiter.get("risk_notes", [])
    }
    return {"debate": debate, "arbiter": arbiter, "summary": summary}


# ============================== 4. 종합 판단 함수 ==============================
def vote_ai(original_image_path: str, new_image_path: str, model: str = MODEL):
    # 1️⃣ 추상화
    ok, abs_res = run_abstraction_agent(original_image_path, new_image_path)
    if not ok:
        raise RuntimeError(f"Abstraction 실패: {abs_res}")

    # 2️⃣ 필터링
    ok, filt_res = run_filtering_agent(abs_res["abstract1"], abs_res["abstract2"])
    if not ok:
        raise RuntimeError(f"Filtering 실패: {filt_res}")

    # 3️⃣ 토론 판정
    panel = run_two_sided_debate(base_path=new_image_path, ref_paths=[original_image_path], rounds=3)
    summary = panel["summary"]
    verdict = summary["verdict"]
    decision = "APPROVE" if verdict == "new" else "DENY"

    return {
        "decision": decision,
        "verdict": verdict,
        "probability": summary["new_work_probability"],
        "metrics": {
            "avg_similarity": summary["avg_similarity"],
            "transformative_degree": summary["transformative_degree"],
            "market_substitution_risk": summary["market_substitution_risk"]
        },
        "reasons": summary.get("reasons", []),
        "risk_notes": summary.get("risk_notes", [])
    }


# ============================== 5. RabbitMQ Consumer ==============================
def on_message(channel, method, properties, body):
    try:
        task = json.loads(body.decode("utf-8"))
        voteId = task["voteId"]
        orig = task["originalImagePath"]
        new = task["derivedImagePath"]

        print(f"[📥] voteId={voteId} 판단 시작")
        result = vote_ai(orig, new)

        response = {
            "voteId": voteId,
            "decision": result["decision"],
            "verdict": result["verdict"],
            "probability": result["probability"],
            "metrics": result["metrics"],
            "reasons": result.get("reasons", []),
            "risk_notes": result.get("risk_notes", [])
        }
        channel.basic_publish(exchange='', routing_key=VOTE_AI_RESPONSE_QUEUE, body=json.dumps(response))
        channel.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[✅] 결과 전송 완료: {voteId} → {result['decision']} ({result['verdict']})")

    except Exception as e:
        print("[❌] on_message 에러:", e)
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
    print("[🚀] 저작권 판단 대기 중...")
    channel.start_consuming()


if __name__ == "__main__":
    client = get_client()
    main()
