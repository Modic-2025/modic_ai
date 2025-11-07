# scenario_with_descriptions_chat.py
from base64 import b64decode
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import requests
import base64

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI


# ──────────────────────────────────────────────────────────────────────────────
# 0) OpenAI 클라이언트
# ──────────────────────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)

client = get_client()
# Chat Completions에서 함수 호출이 잘 되는 모델을 기본값으로 권장
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


IMAGES_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# ──────────────────────────────────────────────────────────────────────────────
# 이미지 생성
# ──────────────────────────────────────────────────────────────────────────────
images_summary: List[List[str]] = []     # e.g., [["img_001.png", "푸른 톤의 풍경"], ...]
chat_summary: str = ""                   # 누적 요약문, 빈 문자열로 시작

def get_image_by_index(index: int):
    """
    히스토리에서 인덱스로 조회.
    - 0 기반 정방향: 0=처음, 1=두 번째, ...
    - 음수 인덱스: -1=마지막(방금), -2=그 전, ...
    """
    n = len(images_summary)
    if n == 0:
        return None
    if -n <= index < n:
        return images_summary[index][0]
    return None


def generate_image_from_text(prompt: str, size: str = "1024x1024") -> Image.Image:
    """
    OpenAI Images API(gpt-image-1)로 텍스트 프롬프트를 보내고
    base64로 받은 이미지를 PIL.Image로 반환
    """
    resp = client.images.generate(
        model=IMAGES_MODEL,
        prompt=prompt,
        size=size,
    )
    b64 = resp.data[0].b64_json   # 그대로 접근 가능
    img = Image.open(BytesIO(b64decode(b64)))
    return img


def edit_image_from_text(
    image_path: str,
    prompt: str,
    size: str = "auto",
    mask_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Image.Image:
    """
    REST로 /v1/images/edits 호출하여 편집 이미지 반환.
    - image_path: 원본 이미지 파일 경로 (PNG 권장, 정사각형)
    - mask_path: 투명 PNG(투명 부분이 '수정 영역') 선택사항
    - size: '256x256' | '512x512' | '1024x1024'
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(image_path, "rb") as f_img:
        files = {"image": (os.path.basename(image_path), f_img, "image/png")}
        if mask_path:
            with open(mask_path, "rb") as f_mask:
                files["mask"] = (os.path.basename(mask_path), f_mask, "image/png")
                resp = requests.post(url, headers=headers, files=files, data={
                    "model": "gpt-image-1", "prompt": prompt, "size": size
                }, timeout=120)
        else:
            resp = requests.post(url, headers=headers, files=files, data={
                "model": "gpt-image-1", "prompt": prompt, "size": size
            }, timeout=120)

    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    return Image.open(BytesIO(base64.b64decode(b64)))


def append_chat_summary(new_line: str, max_len: int = 2000):
    """간단 누적(한 줄씩). 너무 길어지면 앞부분 잘라내기."""
    global chat_summary
    chat_summary = (chat_summary + ("\n" if chat_summary else "") + new_line).strip()
    if len(chat_summary) > max_len:
        chat_summary = chat_summary[-max_len:]


def one_line_desc(text: str, limit: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def download_image_to_tmp(url: str) -> Optional[str]:
    try:
        import uuid, tempfile
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        suffix = ".png"
        tmp_path = os.path.join(tempfile.gettempdir(), f"ref_{uuid.uuid4().hex}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        return tmp_path
    except Exception as e:
        print(f"[경고] 참조 이미지 다운로드 실패: {e}")
        return None


def do_style_transfer():
    print("[Style transfer]: not yet")
    return None
# ──────────────────────────────────────────────────────────────────────────────
# 1) 시나리오 정의
# ──────────────────────────────────────────────────────────────────────────────
SCENARIOS: List[str] = [
    "general_chat",
    "image_task",
]

# ──────────────────────────────────────────────────────────────────────────────
# 2) 시나리오 설명 카탈로그
# ──────────────────────────────────────────────────────────────────────────────
SCENARIO_GUIDE: Dict[str, Dict[str, Any]] = {
    "general_chat": {
        "definition": "일반 대화, 사용법/기능 문의, 설명·조언 등 텍스트 응답이 핵심인 경우",
        "signals": [
            "뭐가 가능해", "어떻게 써", "설명", "도움", "예시 알려줘", "비용", "성능",
            "help", "what can you do", "usage", "docs", "limitations"
        ],
        "examples": [
            "이미지 생성 기능이 뭐가 있어?",
            "스타일 변환은 어떻게 써?",
            "색 바꾸는 거랑 편집의 차이 설명해줘"
        ],
        "anti_examples": [
            "이 그림에서 배경만 어둡게",      # → image_task (edit)
            "사이버펑크 포스터 새로 만들어줘", # → image_task (generate)
            "첫 번째 이미지의 하늘만 파랑으로" # → image_task (recolor_object)
        ],
    },

    "image_task": {
        "definition": "이미지 생성/편집 계열. 하위 subtype으로 세분화하여 실행",
        "subtypes": {
            "generate": {
                "description": "참조 없이 새로 생성",
                "cues": ["새로", "처음부터", "만들어줘", "generate", "create"]
            },
            "edit": {
                "description": "기존 이미지를 편집(배경 흐림, 텍스트 키움, 구성 변경 등)",
                "cues": ["편집", "수정", "바꿔", "edit", "tweak", "change"]
            },
            "recolor_object": {
                "description": "특정 객체의 색상 변경",
                "cues": ["색 바꿔", "컬러 변경", "recolor", "hue", "palette"]
            },
            "style_transfer": {
                "description": "사진/이미지의 전체 화풍 변환",
                "cues": ["스타일 변환", "화풍", "모네 스타일", "style transfer"]
            },
        },
        "signals": [
            # 참조 단서
            "http://", "https://", "#", "별칭", "버전명", "처음", "두 번째", "방금", "지난번",
            # 제작/수정 단서
            "만들어", "그려", "편집", "수정", "합성", "색", "스타일", "recolor", "edit", "generate"
        ],
        "examples": [
            "사이버펑크 도시 포스터 새로 만들어줘",               # generate
            "방금 만든 포스터에서 글자만 키워줘",                # edit (indices = [-1])
            "첫 번째 그림의 하늘을 파란색으로 바꿔줘",           # recolor_object (indices = [0])
            "이 URL 배경에 로고 합성해줘: https://.../bg.png",   # edit (reference_urls)
            "이 사진을 고흐 화풍으로 바꿔줘"                     # style_transfer
        ],
        "anti_examples": [
            "기능 설명해줘", "가격 어때?", "사용법 알려줘"  # → chat
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 3) 시스템 지침
# ──────────────────────────────────────────────────────────────────────────────
def build_system_instructions() -> str:
    return """
너는 '이미지/채팅 라우터'다. 아래 규칙을 지켜라.

[최상위 결정 규칙]
- action은 반드시 하나: {general_chat, image_task}.
- 이미지 작업 신호(생성/편집/색 변경/스타일 변환, URL/별칭/순번/“방금/첫 번째/두 번째” 등)가 있으면 image_task.
- 그 외(기능/사용법/설명/정책/가격/잡담 등)는 general_chat.

[image_task의 subtype 결정]
- generate: 참조(인덱스/URL/별칭) 없이 “새로/만들어/그려/처음부터”.
- edit: 기존 이미지 기반 편집(배경/텍스트/구도 등), 혹은 참조 제공(URL/인덱스/별칭) + 일반적 수정 지시.
- recolor_object: 특정 객체 + 특정 색이 함께 언급되면 우선.
- style_transfer: 화풍/스타일 전환이 핵심이면 우선.

[참조 해석 규칙]
- indices: 사용자가 “방금/지난번/첫 번째/두 번째/n번째”를 말하면 정수 인덱스로 리스트화.
  - 0=첫 번째, 1=두 번째, ...; -1=마지막(가장 최근), -2=그 전 …
  - 여러 개면 모두 담아라(예: [0, -1]).
- reference_urls: 본문 내 URL들을 모두 추출하여 리스트로 제공.
- prompt vs edit_instructions:
  - generate면 prompt에 생성 텍스트를 담고 edit_instructions는 빈 문자열.
  - edit/recolor_object/style_transfer면 edit_instructions에 구체 지시를 담고 prompt는 선택.
- target_objects/target_colors:
  - recolor_object면 객체-색 매핑을 추출하여 각각 리스트로 맞춰 담아라(길이 다르면 가능한 쌍만 사용).
- style_transfer: 화풍 변환 의도가 명확하면 true, 아니면 false.

[출력 스키마]
- action: "general_chat" | "image_task"
- subtype: "generate" | "edit" | "recolor_object" | "style_transfer"   # image_task일 때 필수
- prompt: string | null
- edit_instructions: string | null
- indices: array of integers (기본 빈 배열)
- reference_urls: array of strings (기본 빈 배열)
- target_objects: array of strings (기본 빈 배열)
- target_colors: array of strings (기본 빈 배열)
- style_transfer: boolean (기본 false)
- needs_clarification: boolean
- reason: string(짧게)

[모호성 처리]
- 핵심 정보가 부족하면 needs_clarification=true로 표시하고 reason에 부족한 항목을 적시.
""".strip()

SYSTEM_INSTRUCTIONS = build_system_instructions()

# ──────────────────────────────────────────────────────────────────────────────
# 4) 툴 스키마 (Chat Completions 형식)
# ──────────────────────────────────────────────────────────────────────────────
TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_scenario",
        "description": (
            "이미지 요약본, 시스템 규칙, 채팅 요약, 현재 입력을 바탕으로 "
            "일반 채팅(general_chat)과 이미지 작업(image_task)을 분기한다. "
            "image_task의 subtype은 generate / edit / recolor_object 중 하나이며, "
            "스타일 변환은 style_transfer=true 불린으로 표시한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["general_chat", "image_task"],
                    "description": "최상위 분기"
                },
                "subtype": {
                    "type": "string",
                    "enum": ["generate", "edit", "recolor_object"],
                    "description": "이미지 작업 세부 타입"
                },

                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "히스토리 인덱스 리스트(0=처음…; -1=마지막…)"
                },
                "reference_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "사용자 제공 이미지 URL 목록"
                },

                "prompt": {"type": "string", "description": "이미지 생성/변환 핵심 프롬프트"},
                "edit_instructions": {"type": "string", "description": "편집/채색 지시문(선택)"},

                "target_objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "색 변경/편집 대상 오브젝트"
                },
                "target_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "오브젝트별 목표 색"
                },

                "style_transfer": {
                    "type": "boolean",
                    "description": "스타일 변환 필요 여부"
                },

                "needs_clarification": {"type": "boolean"},
                "reason": {"type": "string"},
                "signals": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["action", "needs_clarification"]
        }
    }
}]



# ──────────────────────────────────────────────────────────────────────────────
# 5) 핸들러 (지금은 print)
# ──────────────────────────────────────────────────────────────────────────────
def respond_general_chat(user_text: str, model: str = MODEL):
    global chat_summary

    # 사람이 읽기 좋은 컨텍스트 블록 (최근 20개 이미지)
    last_items = images_summary[-20:] if len(images_summary) > 20 else images_summary
    if last_items:
        img_lines = [f"- [{i}] {pair[0]} :: {pair[1]}" for i, pair in enumerate(last_items)]
        images_block = "\n".join(img_lines)
    else:
        images_block = "- (없음)"

    context_block = (
        "### 이미지 요약본(최대 20개, 최근 우선)\n"
        f"{images_block}\n\n"
        "### 채팅 요약본\n"
        f"{(chat_summary or '(요약 없음)')}\n"
    )

    system_for_chat = (
        "너는 이 앱의 도우미야. 사용자가 이미지 생성/편집/채색/스타일 변환을"
        " 요청할 수도 있고, 일반 질문을 할 수도 있어. "
        "가능한 경우, 위 컨텍스트를 참고해서 간결하고 실용적으로 답해."
        " 필요 시 다음 행동 예시도 1~2개 제안해줘."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_for_chat},
            {"role": "assistant", "content": context_block},
            {"role": "user", "content": user_text},
        ],
        temperature=0.4,
    )

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        text = "(빈 응답)"

    print("\n[일반 답변]\n" + text)

    # 요약 누적(아주 단순한 형태)
    chat_summary = (chat_summary + ("\n" if chat_summary else "") + f"U: {user_text}\nA: {one_line_desc(text, 200)}").strip()

def execute_image_task(
    *,
    user_text: str,
    subtype: str,
    indices: List[int],
    reference_urls: List[str],
    prompt: Optional[str],
    edit_instructions: Optional[str],
    target_objects: List[str],
    target_colors: List[str],
    style_transfer: bool,
    fig_dpi: int = 100,
):
    """
    route_scenario 결과를 받아 실제 이미지를 생성/편집한다.
    생성/편집 성공 시: 화면 표시 + history/images_summary 업데이트.
    """
    global chat_summary, images_summary

    line = ""
    # 1) subtype 분기
    # 1-1) 이미지 새로 생성
    if subtype == "generate":
        gen_prompt = (prompt or "새 이미지 생성").strip()
        print(f"[생성] prompt={gen_prompt!r}")

        img = generate_image_from_text(gen_prompt, size="1024x1024")
        # 원본 크기로 표시
        w, h = img.size
        plt.figure(figsize=(w/fig_dpi, h/fig_dpi), dpi=fig_dpi)
        plt.axis("off")
        plt.imshow(img)
        plt.show()
        line = f"[생성]: {user_text.strip()}"
    # 1-2) 다른 이미지를 참고해 이미지 생성
    else:
        # === 편집 계열: 원본 이미지 확보 ===
        # 우선순위: indices -> reference_urls -> 실패
        base_path: Optional[str] = None
        if indices:
            # 첫 번째만 사용 (여러 개 편집은 확장 가능)
            sel = get_image_by_index(indices[0])
            if sel:
                base_path = sel

        if not base_path and reference_urls:
            tmp = download_image_to_tmp(reference_urls[0])
            if tmp:
                base_path = tmp

        if not base_path:
            print("[에러] 편집할 원본 이미지를 찾지 못했습니다. indices나 reference_urls를 확인하세요.")
            return

        edit_prompt = (edit_instructions or "").strip()

        if subtype == "recolor_object":
            # edit_instructions 비어있으면 target_objects/colors로 구성
            if not edit_prompt:
                pairs = []
                for i, obj in enumerate(target_objects):
                    color = target_colors[i] if i < len(target_colors) else ""
                    if obj and color:
                        pairs.append(f"{obj}를 {color} 색으로")
                if pairs:
                    edit_prompt = " / ".join(pairs) + " 바꿔줘."

            if not edit_prompt:
                print("[에러] 채색 지시문을 만들 수 없습니다. target_objects/colors를 확인하세요.")
                return

            print(f"[채색 편집] base={base_path}, instr={edit_prompt!r}")
            line = f"[채색 편집]: {user_text.strip()}"

        elif subtype == "edit":
            if not edit_prompt:
                # 최소한 prompt라도 편집 힌트로 사용
                edit_prompt = (prompt or "이미지를 개선해줘").strip()

            print(f"[편집] base={base_path}, instr={edit_prompt!r}")
            line = f"[편집]: {user_text.strip()}"

        else:
            print(f"[에러] 알 수 없는 subtype: {subtype}")
            line = f"[에러]: {user_text.strip()}"
            return

        img = edit_image_from_text(
            image_path=base_path,
            prompt=edit_prompt,
            size="1024x1024",        # 필요 시 동적
            mask_path=None,          # 필요 시 마스크 지원
        )

        w, h = img.size
        plt.figure(figsize=(w/fig_dpi, h/fig_dpi), dpi=fig_dpi)
        plt.axis("off")
        plt.imshow(img)
        plt.show()

    # 2) 스타일 옵션 가미
    if style_transfer:
        do_style_transfer()
        line += "\n style 변환 완료"

    # 3) 저장 & summaries 업데이트
    # 이미지 summary에 저장하는 부분, images_summary 자료형에 맞게 저장
    img_id = f"img_{len(images_summary) + 1:03d}.png"
    if 'img' not in locals() or img is None:
        print("[에러] 이미지 생성/편집에 실패하여 저장을 건너뜁니다.")
        return
    img.save(img_id)

    # images_summary에 저장
    images_summary.append([img_id, line])

    # 채팅 summary를 저장하는 부분, chat_summar 자료형에 맞게 저장, 일부 수정 필요
    append_chat_summary(line)

ACTION_MAP = {
    "general_chat": respond_general_chat,      # 일반 대화/설명/가이드 응답
    "image_task": execute_image_task,  # 이미지 관련 모든 작업(subtype에 따라 내부 분기)
}

# ──────────────────────────────────────────────────────────────────────────────
# 6) 메인: 문자열 입력 → 툴 강제 호출 → 결과 파싱 → 핸들러 실행
# ──────────────────────────────────────────────────────────────────────────────
def classify_and_execute(user_text: str, model: str = MODEL) -> None:
    # 1) images_summary를 사람이 읽기 좋은 블록으로 변환
    #    (최대 20개, 최근 우선 표시 예시)
    last_items = images_summary[-20:] if len(images_summary) > 20 else images_summary
    if last_items:
        img_lines = []
        for i, pair in enumerate(last_items):
            try:
                img_id, desc = pair[0], pair[1]
            except Exception:
                # 형식이 깨졌을 때 방어
                img_id = str(pair)
                desc = ""
            img_lines.append(f"- [{i}] {img_id} :: {desc}")
        images_block = "\n".join(img_lines)
    else:
        images_block = "- (없음)"

    # 2) 컨텍스트 블록 구성
    context_block = (
        "### 이미지 요약본(최대 20개, 최근 우선)\n"
        f"{images_block}\n\n"
        "### 채팅 요약본\n"
        f"{(chat_summary or '(요약 없음)')}\n"
    )

    # 3) 호출 (SYSTEM_INSTRUCTIONS는 그대로 system으로)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant", "content": context_block},  # 컨텍스트 주입
            {"role": "user", "content": user_text},
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "route_scenario"}},
    )

    # 4) 툴 호출 파싱
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        print("[경고] 툴 호출이 감지되지 않음.")
        if msg.content:
            print("[모델텍스트]", msg.content)
        return

    call = tool_calls[0]
    raw = call.function.arguments
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        print("[경고] arguments JSON 파싱 실패:", raw)
        return

    # 5) 결과 해석
    action = args.get("action")
    needs = bool(args.get("needs_clarification", False))
    subtype = args.get("subtype")
    indices = args.get("indices", []) or []
    reference_urls = args.get("reference_urls", []) or []

    prompt = args.get("prompt") or user_text
    edit_instructions = args.get("edit_instructions")
    target_objects = args.get("target_objects", []) or []
    target_colors = args.get("target_colors", []) or []
    style_transfer = bool(args.get("style_transfer", False))

    reason = args.get("reason", "")
    signals = args.get("signals", [])

    # 6) 로깅/라우팅(여기서는 실행기 연결은 생략)
    print(f"[분류] action={action}, subtype={subtype}, needs={needs}, style_transfer={style_transfer}")
    if indices: print(f"[대상 indices] {indices}")
    if reference_urls: print(f"[참조 URL] {reference_urls}")
    if reason: print(f"[이유] {reason}")
    if signals: print(f"[신호] {signals}")

    if action == "general_chat":
        respond_general_chat(user_text, model=model)
        return

    if action == "image_task":
        payload = {
            "user_text": user_text,
            "subtype": subtype,
            "indices": indices,
            "reference_urls": reference_urls,
            "prompt": prompt,
            "edit_instructions": edit_instructions,
            "target_objects": target_objects,
            "target_colors": target_colors,
            "style_transfer": style_transfer,
        }
        execute_image_task(**payload)
        return

    print("[경고] 알 수 없는 action:", action)


# ──────────────────────────────────────────────────────────────────────────────
# 7) 빠른 테스트
# ──────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     samples = [
#         "무슨 기능이 있어?",
#         # "이 URL 이미지에서 사람만 지워줘: https://cdn.example.com/a.png",
#         # "방금 만든 포스터 글자만 키워줘",
#         # "#poster_v2 배경만 어둡게",
#         # "로고랑 이 배경 합쳐서 콜라주로",
#         # "지난번 설정 그대로 해상도만 4K로",
#         "사이버펑크 도시 포스터 새로 만들어줘",
#         "방금 그린 사이버펑크 도시 포스터 그림을 도시가 아니라 시골로 바꿔줘.",
#         "너무 예전 시골 느낌이 나는 것 같아. 조금만 더 현대화해서 그려줘.",
#     ]
#     for s in samples:
#         print(f"\n>>> 입력: {s}")
#         classify_and_execute(s)
#         print("-" * 60)
if __name__ == "__main__":
    print("🖼️ 이미지 작업 시나리오 라우터 (종료하려면 quit 입력)")
    while True:
        s = input("\n>>> 입력: ").strip()
        if not s:
            continue
        if s.lower() in {"quit", "exit", "q"}:
            print("종료합니다.")
            break

        classify_and_execute(s)
        print("-" * 60)