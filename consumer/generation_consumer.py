import json
import uuid
import sqlite3
import time
from contextlib import closing
from pika.exceptions import ChannelClosedByBroker, StreamLostError, AMQPError
from base64 import b64decode
from io import BytesIO
from PIL import Image
import requests
import base64
from typing import List, Dict, Optional
from openai import OpenAI

# Load env variable & Model
from static.rabbitmq import *
from static.model import *
from static.s3 import *
from styletransfer.tasks import wait_for_result


IDEMPOTENT_DB_PATH = os.environ.get("IDEMPOTENT_DB_PATH", "/db/processed.db")
os.makedirs(os.path.dirname(IDEMPOTENT_DB_PATH), exist_ok=True)
with closing(sqlite3.connect(IDEMPOTENT_DB_PATH)) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            request_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def mark_in_progress(request_id: str) -> bool:
    """아직 처리되지 않았으면 기록 후 True, 이미 있으면 False"""
    try:
        with closing(sqlite3.connect(IDEMPOTENT_DB_PATH, timeout=5)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM processed WHERE request_id=?", (request_id,)).fetchone()
            if row and row[0] in ("done", "in_progress"):
                conn.commit()
                return False
            conn.execute("INSERT OR REPLACE INTO processed(request_id, status, updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
                         (request_id, "in_progress"))
            conn.commit()
            return True
    except sqlite3.Error:
        return True  # DB 문제 시에도 처리 진행

def mark_done(request_id: str):
    try:
        with closing(sqlite3.connect(IDEMPOTENT_DB_PATH, timeout=5)) as conn:
            conn.execute("INSERT OR REPLACE INTO processed(request_id, status, updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
                         (request_id, "done"))
            conn.commit()
    except sqlite3.Error:
        pass

def safe_publish(channel, routing_key, body, max_retries=3, sleep_sec=0.5):
    """RabbitMQ publish 재시도"""
    for attempt in range(1, max_retries + 1):
        try:
            channel.basic_publish(exchange='', routing_key=routing_key, body=body)
            return True
        except (ChannelClosedByBroker, StreamLostError, AMQPError, OSError) as e:
            print(f"[경고] publish 실패({attempt}/{max_retries}): {e}")
            time.sleep(sleep_sec)
    return False

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=API_KEY)


def get_s3_key():
    image_name = uuid.uuid4().hex
    extension = "png"
    filename = f"{image_name}.{extension}"
    prefix = S3_PATH_PREFIX.strip('/')
    s3_key = f"{prefix}/{filename}"
    return s3_key, filename, image_name, extension


def upload_to_s3(image_bytes):
    s3_key, filename, image_name, extension = get_s3_key()
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/png"
    )
    return s3_key, filename, image_name, extension


def open_binary(image_path: str):
    print(image_path)
    key = image_path.lstrip("/")
    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)  # <-- Key=key 로!
    data = resp["Body"].read()
    fname = os.path.basename(key)
    ctype = resp.get("ContentType", "image/png")  # S3에 저장한 ContentType 재사용
    return fname, BytesIO(data), ctype


def generate_image_from_text(prompt: str, size: str = "1024x1024") -> Image.Image:
    """
    OpenAI Images API(gpt-image-1)로 텍스트 프롬프트를 보내고
    base64로 받은 이미지를 PIL.Image로 반환
    """
    resp = client.images.generate(
        model=IMAGES_MODEL,
        prompt=prompt,
        size="auto",
    )
    b64 = resp.data[0].b64_json
    img = Image.open(BytesIO(b64decode(b64)))
    return img


def edit_image_from_text(
    image_path: str,
    prompt: str,
    size: str = "1024x1024",
    mask_path: Optional[str] = None,
    reference_image_paths: Optional[List[str]] = None,
    style_image_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Image.Image:
    """
    OpenAI Images API (gpt-image-1) 편집 호출.
    - image_path: 편집의 '베이스' 이미지 (필수)
    - mask_path: 투명 PNG 마스크 (선택)
    - reference_image_paths: 참고 이미지 경로/URL 리스트 (선택)
    - style_image_path: 스타일 가이드 이미지 경로/URL (선택)
    - size: '256x256' | '512x512' | '1024x1024'
    ※ 참고/스타일 이미지는 OpenAI 편집 엔드포인트에서 네이티브 가이드로 쓰이지 않을 수 있음.
      이를 보완하기 위해 prompt에 명시적으로 힌트를 주입하며, 파일 파트는 ref_image_#/style_image로 함께 전송.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {}

    # 필수: base image
    base_name, base_fh, base_ct = open_binary(image_path)
    files["image"] = (base_name, base_fh, base_ct)

    # 선택: mask
    if mask_path:
        mask_name, mask_fh, mask_ct = open_binary(mask_path)
        files["mask"] = (mask_name, mask_fh, mask_ct)

    # 선택: reference images (여러 장)
    ref_list = reference_image_paths or []

    # 참고/스타일 안내를 프롬프트에 주입
    ref_hint = ""
    if ref_list:
        ref_hint += f" 참고이미지 {len([r for r in ref_list if r])}장을 반영해 편집하라."

    effective_prompt = (prompt or "").strip()
    if ref_hint:
        effective_prompt = (effective_prompt + " " + ref_hint).strip()

    data = {
        "model": "gpt-image-1",
        "prompt": effective_prompt,
        "size": size,
    }

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    if not resp.ok:
        print("[OpenAI error payload]", resp.status_code, resp.text)
    resp.raise_for_status()

    # 파일 정리
    base_fh.close()
    if mask_path:
        mask_fh.close()

    b64 = resp.json()["data"][0]["b64_json"]
    return Image.open(BytesIO(base64.b64decode(b64)))


def do_style_transfer(style_image_path, content_image):
    style_name, style_image, style_type = open_binary(style_image_path)
    result_image = wait_for_result(content_image, style_image, prompt=None, preprocessor=None)
    if result_image is None:
        return None

    return result_image


def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. 이번 요청을 기준으로 작업을 결정한다.  
- base: 실제로 수정·변환될 중심 이미지(1개, style_transfer 포함)  
- references: 편집·변환을 돕는 참고 이미지들(0..N, 순서 중요)
- subtype, style_transfer, reason 등 기타 필드도 함께 산출한다.

Ⅰ. 입력 구조 및 참조 규칙
chat에는 **최신 N개의 대화(turn)** 가 저장되어 있다.  
각 대화는 다음 구조를 가진다:

{
  "role": "USER" | "AI",
  "contents": [
    {
      "type": "TEXT" | "IMAGE",
      "text": <문자열 or null>,
      "imagePath": <이미지 경로 or null>,
      "description": <이미지 설명 or null>,
      "fromOriginImage": <bool>   # true면 스타일 변환이 적용된 이미지
    },
    ...
  ]
}

설명:
- `role`이 "USER"면 사용자가 보낸 메시지, "AI"면 AI가 생성한 응답이다.  
- `contents` 배열 안에 여러 요소(TEXT 또는 IMAGE)가 포함될 수 있다.  
- `imagePath`는 실제 이미지 파일 경로이며,  
  `description`은 해당 이미지의 간단한 설명이다.  
- `fromOriginImage=true`면 스타일 변환 결과 이미지임을 의미한다.

chat 내에서 **시간 순서**는 보장되며,
가장 앞의 항목(index 0)이 최신 메시지, 마지막이 가장 오래된 메시지다.

uploads(images_path): 이번 요청 업로드 목록.

대화 내 이미지 지칭 규칙:
- 사용자가 prompt로 요구한 사항(role, 개수)에 맞게 지칭해 아래 규칙에 맞게 base·references 선택하며 불필요한 이미지는 포함하지 않는다.
- 사용자의 “처음/두번째/마지막/방금/이전/최근” 등 표현은 chat에서 이미지만 뽑아서 정해진 요구에 맞게 시간 순서(index)와 role로 자동 매핑. 
  예:
  • “처음” → 가장 오래된 이미지
  • “두번째” → 처음 이미지의 다음 이미지
  • “마지막/방금” → 가장 최신 이미지
  • “네가 만든/AI가 생성한” → role=AI 이미지
  • "내가 보낸" → role=USER 이미지
- “지금 내가 보낸 N개의 이미지” → 최근 USER 업로드 N개를 대상으로 base·references 결정.

Ⅱ. 우선순위  
1) 이번 요청의 prompt·images_path 최우선  
2) 최신 USER 발화만 고려 (단, “처음/두번째/방금” 등 지시 있으면 과거 탐색)  
3) 과거 객체 재사용 금지 (현재 요청만 반영)

Ⅲ. 작업 타입 결정 (R-규칙, 순서 고정)  
R-1 기능 설명 → needs_clarification=true, style_transfer=false, reason=기능 요약  
R0 보류/정지 → needs_clarification=true, style_transfer=false, subtype 없음, reason=""  
R1 업로드만(p 없음, images만) → subtype=style_transfer, style_transfer=true, base=uploads[0] (R-1/R0 제외)  
R2 스타일 전용(스타일 키워드만) → subtype=style_transfer, style_transfer=true  
R3 혼합(스타일+편집 키워드) → subtype=edit, style_transfer=true  
R4 편집(교체/삽입/제거/변경/보정 등) → subtype=edit  
R5 생성(입력 이미지 없음) → subtype=generate  
적용 순서: R-1→R0→R1→R2→R3→R4→R5  

Ⅳ. base 선택 규칙 (결정 알고리즘)  
기본: 업로드 있으면 uploads[0]을 base, 없으면 최신 chat 이미지 사용.  
A. 명시 지시 존재 시 우선:  
　• “AI가 만든/방금 만든” → 최신 role=AI  
　• “내가/방금 보낸/업로드한” → uploads[0] 또는 최신 USER  
　• “chat#k/upload#j/path” → 해당 항목  
B. 명시 지시 없고 uploads 존재 → base=uploads[0]  
C. uploads 없음 → 최신 AI 없으면 최신 USER  
D. 모두 없으면 → needs_clarification=true  

Ⅴ. references 선택 규칙  
- base 제외 후, 편집·변환에 도움되는 이미지만 중요도 순 정렬.  
- 업로드가 있어도 (R1 제외) base는 B 규칙으로 확정되며, 나머지는 references로만 사용.

Ⅵ. 충돌/모호성 해소  
- uploads 없고 base 지시 없을 때 최신 AI와 USER 모두 있으면 최신 AI를 base로.  
- 잠정 USER base라도 같은 맥락의 최신 AI가 있으면 AI로 재결정.  
- “스타일 변환도 적용해줘” 문구 + base 지시 없음 → 최신 AI 결과물을 base로.  

Ⅶ. base / references 표기  
base: { "source": "chat"|"upload", "index": <int> }  
references: [ { "source": ..., "index": ... }, ... ]  
index는 chat_images / uploads의 0-based 인덱스.  

Ⅷ. 지시문 작성  
edit_instructions: “무엇을 어떻게 바꿀지”를 구체적으로.  
references가 있으면 번호로 지칭(예: “references[0] 색감 반영”).  
style_transfer=true면 base만 지정 (화풍 적용은 후처리).  

Ⅸ. clarify 조건  
다음 중 하나면 needs_clarification=true:  
1) chat·uploads 모두 없음  
2) R-1(기능 설명)  
3) R0(보류/정지)  
reason 작성:  
- 일반 clarify → 부족정보·옵션·기본값·예시 포함  
- R-1 → 기능 요약  
- R0 → ""  
R-1/R0 시 subtype 미지정, image_description="".  

Ⅹ. 이미지 설명  
생성될 이미지를 1문장(120자 이내)으로 요약.  
핵심 내용·변환 요소·스타일·질감 등을 간단히 표현.  
예: “라면을 파스타로 바꾼 장면, 따뜻한 조명과 주황 포장.”  
R0일 때 image_description="".  

출력(JSON): subtype, base, references, generate_instructions, edit_instructions, style_transfer, needs_clarification, reason, chat_summary, signals, image_description
""".strip()


SYSTEM_INSTRUCTIONS = build_system_instructions()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_scenario",
        "description": (
            "이번 요청에 대해 작업 타입을 결정하고, base와 references를 '구조화'하여 반환한다. "
            "규칙 요약: "
            "• subtype ∈ {generate, edit, style_transfer} "
            "• edit/style_transfer인 경우, base는 반드시 지정 "
            "• base/references는 {source, index?, path?} 구조로 지정 "
            "• source ∈ {chat, upload}. chat이면 index로 chat_images[i]를 가리키는 것을 우선, "
            "  upload면 index로 uploads[j]를 가리키는 것을 우선(가능하면 path는 비워둔다). "
            "• 별다른 지시가 없으면 base=가장 최신 chat 이미지(chat#max i). "
            "• references에는 base를 절대 포함하지 않는다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtype": {
                    "type": "string",
                    "enum": ["generate", "edit", "style_transfer"],
                    "description": "이미지 작업 세부 타입"
                },

                "base": {
                    "type": "object",
                    "description": (
                        "실제 편집/변환할 대상 1개. "
                        "source=chat이면 index(정수)로 chat_images[index]를 가리켜라. "
                        "source=upload이면 index(정수)로 uploads[index]를 가리켜라. "
                        "가능하면 path는 비우고 index만 사용한다."
                    ),
                    "properties": {
                        "source": {"type": "string", "enum": ["chat", "upload"]},
                        "index": {"type": "integer", "description": "chat_images/uploads의 인덱스"},
                        "path":  {"type": "string", "description": "인덱스가 없을 때만 S3 Key/URL"}
                    },
                    "required": ["source"]
                },

                "references": {
                    "type": "array",
                    "description": (
                        "참고 이미지들(0..N). base는 절대 포함하지 않는다. "
                        "각 항목은 base와 동일 구조. 중요도 순서대로 나열(0이 가장 중요)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "enum": ["chat", "upload"]},
                            "index": {"type": "integer"},
                            "path":  {"type": "string"}
                        },
                        "required": ["source"]
                    }
                },

                "generate_instructions": {
                    "type": "string",
                    "description": "subtype=generate일 때의 생성 프롬프트(구체적)"
                },
                "edit_instructions": {
                    "type": "string",
                    "description": "subtype=edit일 때의 편집 지시문(구체적)"
                },

                "style_transfer": {
                    "type": "boolean",
                    "description": "스타일 변환 필요 여부"
                },

                "image_description": {
                    "type": "string",
                    "description": "결과물 설명(후속 참조용)"
                },

                "needs_clarification": {
                    "type": "boolean",
                    "description": (
                        "추가 정보 필요 여부. "
                        "edit/style_transfer인데 base를 특정할 수 없을 때 반드시 true"
                    )
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "needs_clarification이 true일 때만. 한국어로, "
                        "부족한 정보 + 바로 고를 수 있는 3~5개 옵션 + 안전한 기본값/근거 + 예시 답변 한 줄"
                    )
                },

                "signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "탐지된 키워드/신호(디버깅용)"
                },
                "chat_summary": {
                    "type": "string",
                    "description": "대화 요약(최신 기준)"
                }
            },
            "required": ["subtype", "needs_clarification"]
        }
    }
}]

def execute_image_task(
    *,
    prompt: Optional[str],
    subtype: str,
    base_path: Optional[str],            # NEW
    extra_refs: List[str],               # NEW
    generate_instructions: Optional[str],
    edit_instructions: Optional[str],
    style_transfer: bool,
    style_image_path: Optional[str] = None,
) -> (bool, str, object):
    def _pil_to_bytesio(img: Image.Image) -> BytesIO:
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _bytesio_to_pil(fp: BytesIO) -> Image.Image:
        fp.seek(0)
        im = Image.open(fp)
        return im.convert("RGB") if im.mode != "RGB" else im

    # 출력 경로 준비
    os.makedirs("outputs", exist_ok=True)
    existing = [f for f in os.listdir("outputs") if f.startswith("img_") and f.endswith(".png")]
    out_path = os.path.join("outputs", f"img_{len(existing)+1:03d}.png")

    print("Subtype: ", subtype)
    # 생성
    if subtype == "generate":
        gen_text = (generate_instructions or prompt or "").strip()
        if not gen_text:
            print("[에러] generate 프롬프트가 비어 있습니다.")
            return False, f"[에러] generate 프롬프트가 비어 있습니다.", None
        print(f"[생성] prompt={gen_text!r}")
        try:
            img = generate_image_from_text(gen_text, size="1024x1024")
            img.save(out_path)
            print(f"[완료] 생성 이미지 저장: {out_path}")

            # 스타일 변환
            if style_transfer and style_image_path:
                content_fp = _pil_to_bytesio(img)
                result_fp = do_style_transfer(style_image_path, content_fp)
                print(f"[정보] style_transfer=True, style_image_path={style_image_path}")
                if result_fp is None:
                    return False, f"[이미지 생성 단계, 스타일 변환 에러]", None
                img = _bytesio_to_pil(result_fp)

            return True, "", img
        except Exception as e:
            print(f"[에러]: {e}")
            return False, f"[에러]: {e}", None

    # 편집 지시문
    edit_text = (edit_instructions or "").strip()
    if not edit_text:
        edit_text = "이미지를 개선해줘"

    print(f"[편집] base={base_path}, refs={extra_refs}, instr={edit_text!r}")
    try:
        if subtype in ("edit", "style_transfer") and not base_path:
            return False, "[에러] base_path가 비었습니다.", None

        if subtype == "edit":
            img = edit_image_from_text(
                image_path=base_path,
                prompt=edit_text,
                size="auto",
                mask_path=None,
                reference_image_paths=extra_refs,
                style_image_path=None,
            )

            # 스타일 변환
            if style_transfer:
                if not style_image_path:
                    return False, "[에러] 스타일 변환 요청이지만 style_image_path가 없습니다.", None
                content_fp = _pil_to_bytesio(img)
                result_fp = do_style_transfer(style_image_path, content_fp)
                print(f"[정보] style_transfer=True, style_image_path={style_image_path}")
                if result_fp is None:
                    return False, f"[스타일 변환 에러]", None
                img = _bytesio_to_pil(result_fp)

        elif subtype == "style_transfer":
            if not style_image_path:
                return False, "[에러] 스타일 변환 요청이지만 style_image_path가 없습니다.", None
            _, content_fh, _ = open_binary(base_path)
            content_fh.seek(0)
            result_fp = do_style_transfer(style_image_path, content_fh)
            if result_fp is None:
                return False, "[스타일 변환 에러]", None
            img = _bytesio_to_pil(result_fp)

        else:
            return False, f"[Image task subtype Error: {subtype}]", None

    except Exception as e:
        print(f"[에러] 편집 실패: {e}")
        return False, f"[에러] 편집 실패: {e}", None

    img.save(out_path)
    print(f"[완료] 편집 이미지 저장: {out_path}")

    return True, "", img


def classify_and_execute(
    prompt: str,
    images_path: list,
    style_image_id: str,
    style_image_path: str,
    recent_chat: list,
    chat_summary: str,
    model: str = MODEL,
):
    """
    텍스트와 이미지를 '같은 메시지'의 content 배열로 섞어 전달.
    - chat 내 텍스트/이미지를 turn 순서대로 넣고,
    - uploads(images_path)는 별도 섹션으로 이어서 넣음.
    - 각 이미지는 라벨(chat#i / upload#j)을 텍스트로 먼저 명시하고, 바로 다음에 image_url 로 실제 이미지를 첨부.
    - TOOL은 indices(=chat 이미지 인덱스만), reference_urls(file://, chat/업로드 모두)에 맞춰 응답.
    """

    def _safe(s):
        return (s or "").replace("\n", " ").strip()

    def _bool(v):
        if v is True or v is False:
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("true", "1", "yes", "y"): return True
            if t in ("false", "0", "no", "n"): return False
        raise ValueError(f"bool 값이 잘못됨 {v}")

    def _is_http(u: str) -> bool:
        return isinstance(u, str) and u.startswith("https://")

    def _json_text_block(obj: dict):
        return {
            "type": "text",
            "text": json.dumps(obj, ensure_ascii=False)
        }

    def _resolve_item(item, chat_image_map, uploads):
        """item: {source: 'chat'|'upload', index?:int, path?:str} -> 실제 경로(str)"""
        if not item or "source" not in item:
            return None
        src = item["source"]
        idx = item.get("index", None)
        pth = item.get("path", None)

        if src == "chat":
            if isinstance(idx, int) and (idx in chat_image_map):
                return chat_image_map[idx]
            return pth  # (fallback)
        if src == "upload":
            if isinstance(idx, int) and 0 <= idx < len(uploads):
                return uploads[idx]
            return pth  # (fallback)
        return None

    # 1) 간단한 설명과 사용자 프롬프트 (비어있으면 빈 문자열로)
    content = [{
        "type": "text",
        "text": (
            "아래는 하나의 JSON 기반 대화 컨텍스트입니다.\n"
            "- prompt: 이번 요청의 사용자 텍스트\n"
            "- chat_images: 과거 대화 중 이미지 목록 (chat#i)\n"
            "- uploads: 이번 요청에 포함된 업로드 이미지 목록 (S3 Key 또는 URL)\n"
            "- chat_summary: 이전 대화의 요약 (선택)\n"
            "⚠ 모든 항목은 JSON 객체로 제공되며, 사람 읽기용 텍스트는 포함되지 않습니다.\n"
            "툴은 prompt / chat_images / uploads를 기반으로 base 및 references를 결정해야 합니다."
        )
    }, _json_text_block({
        "type": "prompt",
        "value": _safe(prompt)
    })]

    # 2) recent_chat에서 이미지만 구조화 (i 인덱스는 chat#i와 동일하게 부여)
    chat_image_map: Dict[int, str] = {}
    chat_images = []
    img_counter = 0

    for turn in list(recent_chat or []):
        role = turn.get("role", "user")
        for c in turn.get("contents", []):
            ctype = (c.get("type") or "").lower()
            if ctype == "image":
                img_path = _safe(c.get("imagePath", ""))
                desc = _safe(c.get("description", ""))
                from_origin_image = _bool(c.get("fromOriginImage"))
                if not img_path:
                    continue
                chat_images.append({
                    "i": img_counter,  # ← indices[0]로 고를 때 사용할 정수 인덱스
                    "path": img_path,  # ← S3 키 or https URL (그대로 사용)
                    "role": role,  # user/assistant
                    "desc": desc,  # 선택 설명
                    "fromOriginImage": from_origin_image  # bool
                })
                chat_image_map[img_counter] = img_path
                img_counter += 1

    # chat 이미지 목록을 JSON 블록으로 전달
    content.append(_json_text_block({
        "type": "chat_images",
        "value": chat_images  # []일 수 있음
    }))

    # 3) 업로드 목록을 JSON 블록으로 전달 (사람용 라벨 제거)
    uploads = [_safe(p) for p in (images_path or [])]
    content.append(_json_text_block({
        "type": "uploads",
        "value": uploads  # 예: ["ai-request/...."] 또는 []
    }))

    # (선택) chat_summary도 기계가 읽기 쉽게 JSON으로만 전달
    if chat_summary:
        content.append(_json_text_block({
            "type": "chat_summary",
            "value": _safe(chat_summary)
        }))

    # 디버그 로그도 JSON만 찍기
    print({
        "prompt": _safe(prompt),
        "chat_images_count": len(chat_images),
        "uploads": uploads
    })

    # ── 5) route_scenario 호출: 텍스트+이미지 함께 전달
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": content},  # 한 메시지에 text+image 동시 포함
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "route_scenario"}},
        temperature=0.2,
    )
    # ── 6) 툴 아웃풋 파싱
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        print("[경고] 툴 호출이 감지되지 않음.")
        if msg.content:
            print(f"[모델텍스트]: {msg.content}")
        return "error", f"[모델텍스트]: {msg.content}"

    call = tool_calls[0]
    raw = call.function.arguments
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        print(f"[에러] arguments JSON 파싱 실패: {raw}")
        return "error", f"[에러] arguments JSON 파싱 실패: {raw}"

    new_chat_summary = args.get("chat_summary", chat_summary)

    # ── 7) 결과 해석 (이미지 작업만 수행)
    needs = bool(args.get("needs_clarification", False))
    reason = args.get("reason", "")
    if needs:
        print(f"추가적인 설명 필요, 이유: {reason}")
        message = {"response": f"{reason}", "chat_summary": new_chat_summary, "reason": reason}
        return "clarify", message

    subtype = args.get("subtype")  # "generate" | "edit" | "style_transfer"

    # NEW: base/references 구조 해석 유틸
    uploads = [_safe(p) for p in (images_path or [])]  # 이미 위에서 만든 값과 동일 개념

    # NEW: base / references 해석
    base_obj = args.get("base")
    base_path = _resolve_item(base_obj, chat_image_map, uploads)

    ref_objs = args.get("references", []) or []
    extra_refs = []
    for r in ref_objs:
        rp = _resolve_item(r, chat_image_map, uploads)
        if rp:
            extra_refs.append(rp)

    generation_prompt = args.get("generate_instructions") or prompt
    edit_instructions = args.get("edit_instructions")
    style_transfer = bool(args.get("style_transfer", False))
    image_description = args.get("image_description", "")

    # 필수 검증: 편집/스타일 변환이면 base 필수
    if subtype in ("edit", "style_transfer") and not base_path:
        print("[경고] 편집/스타일 변환인데 base 미지정 → clarify로 전환")
        message = {"response": "편집/스타일 변환인데 base 이미지를 특정하지 못했습니다.",
                   "chat_summary": new_chat_summary,
                   "reason": "base가 비었습니다. 최근 업로드 또는 최신 USER 이미지를 base로 사용할지 선택해 주세요."}
        return "clarify", message

    print(f"[분류] action=image task(고정), subtype={subtype}, needs={needs}, style_transfer={style_transfer}")
    print(f"[대상 base] {base_path}")
    if extra_refs:
        print(f"[참조 refs] {extra_refs}")

    # ── 8) 이미지 작업 실행
    payload = {
        "prompt": prompt,
        "subtype": (subtype or "generate"),
        "base_path": base_path,  # NEW
        "extra_refs": extra_refs,  # NEW
        "generate_instructions": (generation_prompt if subtype == "generate" else None),
        "edit_instructions": (edit_instructions if (subtype != "generate") else None),
        "style_transfer": style_transfer,
        "style_image_path": style_image_path,
    }

    success, message, img = execute_image_task(**payload)
    if not success:
        return "error", message
    try:
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        s3_key, file_name, image_name, _ = upload_to_s3(buf.getvalue())
        message = {"image_path": s3_key, "file_name": file_name, "image_name": image_name, "description": image_description, "style_transfer": style_transfer,
                   "chat_summary": new_chat_summary}
        return "ok", message
    except Exception as e:
        print(e)
        return "error", e

def on_message(channel, method, properties, body):
    try:
        raw_body = body.decode("utf-8")
        print("[📥] 작업 수신:", raw_body)
        task = json.loads(raw_body)

        request_id = task.get("requestId")
        if not request_id:
            print("[경고] requestId 없음 → DLX로 이동")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        # [PATCH] 멱등성 체크
        if not mark_in_progress(request_id):
            print(f"[멱등] 이미 처리된 요청 {request_id} → ACK 후 스킵")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        prompt = task.get("prompt", "")
        images_path = task.get("imagesPath", [])
        style_image_id = task.get("styleImageId", "")
        style_image_path = task.get("styleImagePath", "")
        recent_chat = task.get("chat", [])
        chat_summary = task.get("chatSummary", "")

        success, message = classify_and_execute(
            prompt, images_path, style_image_id, style_image_path, recent_chat, chat_summary
        )

        # [PATCH] 상태별 응답 처리
        if success == "ok":
            resp = {
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
            if safe_publish(channel, IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, json.dumps(resp)):
                mark_done(request_id)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                print("[에러] publish 실패 → DLX로 이동")
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        elif success == "clarify":
            resp = {
                "isSuccess": True,
                "requestId": request_id,
                "isImageGenerated": False,
                "textContext": message["reason"],
                "chatSummary": message["chat_summary"]
            }
            if safe_publish(channel, IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, json.dumps(resp)):
                mark_done(request_id)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        else:
            print(f"[에러] 처리 실패: {success} / {message}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as e:
        print(f"[❌] on_message 예외: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    import ssl
    import pika
    import time

    while True:
        connection = None
        channel = None
        try:
            context = ssl.create_default_context()
            credentials = pika.PlainCredentials(IMAGE_GENERATION_CHAT_USERNAME, IMAGE_GENERATION_CHAT_PASSWORD)
            params = pika.ConnectionParameters(
                host=IMAGE_GENERATION_CHAT_HOST,
                port=int(IMAGE_GENERATION_CHAT_PORT),
                credentials=credentials,
                ssl_options=pika.SSLOptions(context),
                heartbeat=120,
                blocked_connection_timeout=300,
                client_properties={"connection_name": "image-consumer"},
            )
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.confirm_delivery()

            channel.queue_declare(
                queue=IMAGE_GENERATION_CHAT_QUEUE,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "ai.image.request.dlx",
                    "x-dead-letter-routing-key": "ai.image.request.retry"
                }
            )

            channel.basic_consume(
                queue=IMAGE_GENERATION_CHAT_QUEUE,
                on_message_callback=on_message,
                auto_ack=False
            )

            print("[🚀] 이미지 생성 작업 대기 중...")
            channel.start_consuming()

        except KeyboardInterrupt:
            print("[🧩] 사용자 종료 요청")
            break
        except Exception as e:
            print(f"[경고] 소비자 루프 예외 발생: {e}")
            time.sleep(2.0)
            continue
        finally:
            if channel and channel.is_open:
                try:
                    channel.stop_consuming()
                except:
                    pass
            if connection and not connection.is_closed:
                connection.close()
            print("[✔] 연결 종료")


if __name__ == "__main__":
    client = get_client()
    main()
