import pika
import ssl
import json
import uuid
# import botocore.exceptions

# from io import BytesIO

# from styletransfer.tasks import wait_for_result

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

# Load style transfer model
from styletransfer.tasks import wait_for_result


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


# ──────────────────────────────────────────────────────────────────────────────
# 1) 세부 이미지 생성/수정/변환 함수
# ──────────────────────────────────────────────────────────────────────────────
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

    # def _open_binary(path_or_url: str):
    #     # path_or_url이 http(s)/file://면 가져오고, 아니면 로컬 파일 오픈
    #     if path_or_url.startswith(("http://", "https://")):
    #         r = requests.get(path_or_url, timeout=30)
    #         r.raise_for_status()
    #         # 파일명 유추
    #         fname = os.path.basename(path_or_url.split("?")[0]) or "ref.png"
    #         return (fname, BytesIO(r.content), "image/png")
    #     else:
    #         raise ValueError(f"path_or_url 값이 잘못됨 {path_or_url}")
    # def _open_binary(image_path: str):
    #     key = image_path.lstrip("/")
    #     resp = s3_client.get_object(Bucket=S3_BUCKET, Key=image_path)
    #     data = resp["Body"].read()
    #     fname = os.path.basename(key)
    #     return fname, BytesIO(data), f"image/{fname.split('.')[-1].lower().replace('jpg','jpeg')}"

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
    # for i, ref in enumerate(ref_list):
    #     if not ref:
    #         continue
    #     try:
    #         rn, rf, rct = open_binary(ref)
    #         # 서버가 인식하면 활용, 무시해도 안전
    #         files[f"ref_image_{i}"] = (rn, rf, rct)
    #     except Exception as e:
    #         print(f"[경고] reference 이미지 로드 실패({ref}): {e}")

    # # 선택: style image
    # if style_image_path:
    #     try:
    #         sn, sf, sct = open_binary(style_image_path)
    #         files["style_image"] = (sn, sf, sct)
    #     except Exception as e:
    #         print(f"[경고] style 이미지 로드 실패({style_image_path}): {e}")

    # 참고/스타일 안내를 프롬프트에 주입
    ref_hint = ""
    if ref_list:
        ref_hint += f" 참고이미지 {len([r for r in ref_list if r])}장을 반영해 편집하라."
    # if style_image_path:
    #     ref_hint += " style_image의 화풍/질감/톤을 참고하라."

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


# ──────────────────────────────────────────────────────────────────────────────
# 1) 시스템 지침 (베이스/참고 이미지 선택 및 edit 프롬프트 생성 규칙 명시)
# ──────────────────────────────────────────────────────────────────────────────
# 너는 '이미지 편집 플래너'다. 사용자의 최신 요청을 기준으로 **작업 타입**, **base(편집 대상 1개)**, **references(참고 N개, 순서 중요)**를 결정해 결과를 반환한다.
#
# [chat_summary]
# 채팅 요약은 단순 참고용으로만 사용
#
# [1) 작업 타입 결정 – 우선순위]
# - R1. 기본적으로 스타(텍스트 없이) 업로드만 있음 → subtype=style_transfer, style_transfer=true, base=업로드.
# - R2. 최신 USER 텍스트가 ‘스타일/화풍/그림체/style’만 포함(편집 키워드 없음) → subtype=style_transfer, style_transfer=true.
# - R3. 최신 USER 텍스트에 ‘스타일 변환’과 편집 키워드(교체/합성/삽입/제거/옷/배경/수정/변경/들고 등)가 함께 있음 → subtype=edit, style_transfer=true.
# - R4. 그 외: 일부 요소 수정/합성/교체/삽입/제거/부분 편집이면 subtype=edit. 입력 이미지 전혀 없고 새로 그려야 하면 subtype=generate.
# ※ 항상 “최신 USER 발화 우선”. 과거에 edit 맥락이 있어도 최신 발화가 R1/R2면 style_transfer가 우선.
#
# [2) base / references 선택]
# - base: 실제로 수정/변환될 중심 이미지 단 1개. 별다른 지칭이 없는 경우 최근 이미지가 Base(**중요**)
# - references: base 편집을 위한 참고 이미지들(의미 있는 우선순서대로 나열; 0번이 가장 중요).
# - 지칭 해석:
#   • “A를 B처럼/로 바꿔줘” → base=A, references[0]=B
#   • “네가(너가) 생성한 이미지” → 가장 최근 AI 이미지, role:AI
#   • “내가/방금 보낸/올린 이미지” → 가장 최근 USER 이미지, role:USER
# - 둘 다 언급되면 “수정 대상”을 base, 나머지 비교/참고 대상을 references로.
#
# [3) indices / reference_urls 기입]
# - base가 chat 이미지면: indices[0] = (chat#i의 i). reference_urls에 base는 넣지 않는다.
# - base가 uploads(images_path)이면: indices=[] 로 두고 reference_urls[0] = uploads[0] (핸들러가 이를 base로 사용).
# - references에는 항상 base를 제외하고, 나머지 참고 이미지를 순서대로 넣는다(S3 키/URL 그대로, 검증/변환 금지).
#
# [4) 프롬프트 작성]
# - edit_instructions: “무엇은 유지 / 무엇을 어떻게 바꿀지”를 구체적으로. reference가 있는 경우 references의 번호를 지칭.
# - style_transfer=true가 함께 요구되면 화풍 적용은 후처리(핸들러 처리)로 가정. 스타일 옵션 재질문 금지(기본 스타일로 진행).
#
# [5) clarify]
# - base/references를 전혀 특정할 수 없을 때만 needs_clarification=true.
# - R1·R2 상황에서는 clarify 금지.
#
# [signals]
# - 판단에 기여한 키워드 반환
# [6) 출력]
# - subtype, edit_instructions, indices, reference_urls, style_transfer, needs_clarification, reason, chat_summary, signals
def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. **이번 요청**을 기준으로 작업을 결정하고,
- **base**: 실제로 수정/변환될 중심 이미지(정확히 1개, ★style_transfer 포함★)
- **references**: 편집/변환을 돕는 참고 이미지들(0..N, **순서 중요**)
- 기타 필드를 산출한다.

[원칙 0: 우선순위 신호 (매우 중요)]
1) **이번 요청의 필드가 최우선**이다.
   - 이번 요청의 `prompt`, `images_path`가 있으면 과거 대화보다 우선한다.
2) "최신 USER 발화"는 **가장 마지막 USER 턴**만을 의미한다.
   - 예외: 그 발화에 "처음/두번째/방금 네가/내가 보낸" 등 **지시어**가 있으면,
     지시어 해석을 위해 필요한 범위만 과거 턴을 조회한다.
3) **과거에 언급된 객체명/지시문은 재사용 금지**. 현재 발화와 references에서만 명사/객체를 추출한다.

[1) 작업 타입 결정 – 규칙(R)과 적용 순서]
- **R1. 업로드만**: 이번 요청이 `prompt`가 비었고 `images_path`만 존재 →
  `subtype=style_transfer`, `style_transfer=true`, **base=uploads[0]**.
- **R2. 스타일 전용**: 최신 USER 텍스트가 ‘스타일/화풍/그림체/style/스타일 변환’ 등
  **스타일 키워드만** 포함(편집 키워드 없음) →
  `subtype=style_transfer`, `style_transfer=true`.
- **R3. 혼합**: 최신 USER 텍스트에 **스타일 키워드 + 편집 키워드**(교체/합성/삽입/제거/변경/수정/배경/옷/들고 등)가 **함께** 존재 →
  `subtype=edit`, `style_transfer=true` (편집 후 스타일 적용).
- **R4. 편집**: 요소의 교체/합성/삽입/제거/부분 수정/레이아웃 보정이 요구되면 →
  `subtype=edit`.
- **R5. 생성**: 입력 이미지 없이 새로 그려야 하면 →
  `subtype=generate`.

※ 항상 **R1→R2→R3→R4→R5** 순서로 판정한다.

[2) base / references 선택 규칙 (★필수★)
— "바로 직전 이미지" 기본값을 사용자(UPLOAD/USER)로 고정]
- **base**: 실제로 손댈/변환할 이미지 1개. `edit`와 `style_transfer` 모두 **반드시 지정**.
- ★ 기본 선정 우선순위(명시 지시가 없을 때):
  1) `images_path`가 **비어있지 않으면** → **base = uploads[0]** (최우선)
  2) 아니면 **chat_images 중 가장 최근 `role=USER` 이미지**
  3) ★ `role=AI` 이미지는 **사용자가 명시적으로 "네가 생성한/AI 이미지"라고 지칭한 경우에만** base 후보로 허용
     (그 외에는 **기본값에서 절대 선택 금지**)
  4) 위 모두 없으면 → `needs_clarification=true`
- 지칭 해석 예:
  • “A를 B처럼/로 바꿔줘” → base=A, references[0]=B
  • “네가(너가) 생성한 이미지” → ★가장 최근 **AI** 이미지(명시 지시가 있을 때만)
  • “내가/방금 보낸/올린 이미지” → 가장 최근 **USER** 이미지
- **references**: base 편집/변환을 위한 참고 N개. **의미 있는 우선순서**로 정렬(0번이 가장 중요).
  - base는 references에 절대 포함하지 않는다.

[3) indices / reference_urls 채우기 (★반드시 base를 표현★)]
- base가 **chat 이미지**면: `indices[0] =` 그 이미지의 chat 인덱스(i).
  `reference_urls`에는 **base를 넣지 않는다**.
- base가 **uploads(images_path)**면:
  - `indices=[]`로 두고 **`reference_urls[0] = uploads[0]`**  ← 핸들러가 이것을 **base**로 사용한다(특수 규칙).
- references에는 항상 **base를 제외**하고, 참고 이미지만 **순서대로** 넣는다.
- **중요**: `subtype`이 `edit` 또는 `style_transfer`인 경우,
  **반드시 `indices` 또는 `reference_urls[0]` 중 하나로 base를 지정**해야 한다.
  (둘 다 비우지 말 것. 비울 경우 `needs_clarification=true`로 전환.)

[4) 지시문 작성]
- `edit_instructions`: “무엇은 유지 / 무엇을 어떻게 바꿀지”를 **짧고 구체적으로**.
  references가 있으면 **번호로 지칭**(예: “references[0]의 질감/색을 반영…”).
- `style_transfer=true`인 경우:
  - 플래너는 **base만 정확히 지정**한다.(화풍/스타일 적용은 후처리이므로 별도 편집 지시 불필요)
  - 스타일 가이드를 참고 이미지로 제공해야 한다면,
    `references[0]`에 스타일 가이드 이미지를 넣고, `edit_instructions`에 “references[0]의 화풍/톤”을 짧게 기술.

[5) clarify (질의 필요 조건)]
- ★ `images_path`도 비었고, chat에도 **USER 이미지가 전혀 없으며** 사용자가 "AI가 만든"이라고 지칭하지 않은 경우에만 `needs_clarification=true`.
- 이유(`reason`)는 한국어로, 다음을 반드시 포함:
  1) 무엇이 부족한지
  2) 사용자가 바로 선택할 3~5개 옵션(번호 목록)
  3) 진행 가능한 안전한 기본값 제안과 근거
  4) 그대로 복붙 가능한 예시 답변 한 줄

[signals]
- 판정에 기여한 **키워드/지시어**를 배열로 반환(예: ["스타일 변환","교체","로켓"]).

[6) 출력]
- `subtype`, `edit_instructions`, `indices`, `reference_urls`,
  `style_transfer`, `needs_clarification`, `reason`, `chat_summary`, `signals`
""".strip()


SYSTEM_INSTRUCTIONS = build_system_instructions()

# ──────────────────────────────────────────────────────────────────────────────
# 2) 툴 스키마 (Chat Completions 형식) — 필드 추가 없이 설명 강화
# ──────────────────────────────────────────────────────────────────────────────
TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_scenario",
        "description": (
            "시스템 규칙과 **최신 USER 발화 우선 원칙**에 따라 작업 타입을 결정한다. "
            "규칙: ① 텍스트 없이 업로드만 있으면 style_transfer, "
            "② 최신 발화가 '스타일/화풍/그림체/style'만 포함하면 style_transfer, "
            "③ '스타일 변환+편집 키워드'가 함께면 edit + style_transfer=true. "
            "base가 chat 이미지면 indices[0]로 지정, base가 uploads면 indices는 비우고 reference_urls[0]에 uploads[0]을 넣는다. "
            "references에는 base를 절대 넣지 말고, 참고 우선순서대로 나열한다. "
            "edit/style이면 edit_instructions를 구체적으로 작성한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtype": {
                    "type": "string",
                    "enum": ["generate", "edit", "style_transfer"],
                    "description": "이미지 작업 세부 타입(스타일 변환은 style_transfer=true), 무조건 하나는 지정해야 됨."
                },
                "reference_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "이미지 편집 시 참고할 이미지 목록. **http(s) URL 또는 S3 Key** 그대로 넣기(검증/변환 금지)."
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "chat 이미지 선택 시: indices[0] = image chat#i의 i (정수). **-1 사용 금지**. i는 0부터 시작."
                },
                "generate_instructions": {"type": "string", "description": "이미지 '생성' 프롬프트(구체적으로)"},
                "edit_instructions": {"type": "string", "description": "최대한 사용자의 prompt에 맞춰 편집 지시문"},

                "image_description": {
                    "type": "string",
                    "description": "생성할 이미지에 대한 설명을 반환합니다. 이 설명은 나중에 이미지에 대해서 참고할 때 쓰입니다."
                },

                "style_transfer": {
                    "type": "boolean",
                    "description": "스타일 변환 필요 여부(true면 style transfer)"
                },

                "needs_clarification": {"type": "boolean", "description": "추가 정보 필요 여부"},
                "reason": {
                    "type": "string",
                    "description":
                        "needs_clarification일 때 **한국어로** 작성. 반드시 포함: "
                        "1) 부족한 정보가 무엇인지, "
                        "2) 사용자가 바로 선택할 3~5개 옵션(번호 목록), "
                        "3) 진행 가능한 안전한 기본값 제안과 근거, "
                        "4) 그대로 복붙 가능한 예시 답변 한 줄. "
                        "무성의한 '빈 프롬프트' 같은 문구 금지. 사용자 관점에서 친절하고 구체적으로."
                },
                "signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "탐지된 키워드/신호(디버깅용)"
                },
                "chat_summary": {"type": "string", "description": "지금까지의 채팅을 요약한 글. 최신 채팅을 기준으로 자세하게 정리."}
            },
            "required": ["needs_clarification"]
        }
    }
}]


# ──────────────────────────────────────────────────────────────────────────────
# 3) 핸들러
# ──────────────────────────────────────────────────────────────────────────────
def execute_image_task(
    *,
    prompt: Optional[str],
    subtype: str,
    indices: List[int],
    reference_urls: List[str],
    generate_instructions: Optional[str],
    edit_instructions: Optional[str],
    style_transfer: bool,
    style_image_path: Optional[str] = None,
    chat_image_map: Optional[Dict[int, str]] = None,
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
    chat_image_map = chat_image_map or {}

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

    base_path = None
    if indices:
        cand = chat_image_map.get(indices[0])
        if cand:
            base_path = cand

    if not base_path and reference_urls:
        base_path = reference_urls[0]

    if not base_path:
        return False, "[에러] 편집 base 이미지를 찾지 못했습니다.", None

    # 참고 이미지는 base를 제외한 나머지
    extra_refs = reference_urls[1:] if reference_urls and len(reference_urls) > 1 else []

    # 편집 지시문
    edit_text = (edit_instructions or "").strip()
    if not edit_text:
        edit_text = "이미지를 개선해줘"

    print(f"[편집] base={base_path}, refs={extra_refs}, instr={edit_text!r}")
    try:
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


# ──────────────────────────────────────────────────────────────────────────────
# 4) 메인: 문자열 입력 → 툴 강제 호출 → 결과 파싱 → 핸들러 실행
# ──────────────────────────────────────────────────────────────────────────────
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

    def _bool(s):
        if s != True and s != False:
            raise ValueError(f"bool 값이 잘못됨 {s}")
        return s

    def _is_http(u: str) -> bool:
        return isinstance(u, str) and u.startswith("https://")

    def _json_text_block(obj: dict):
        return {
            "type": "text",
            "text": json.dumps(obj, ensure_ascii=False)
        }

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

    for turn in (recent_chat or []):
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

    # 선택 결과 (정화(sanitize) 포함)  # ★
    raw_indices = args.get("indices", []) or []  # chat 이미지 선택 시: indices[0] = i
    raw_refs = args.get("reference_urls", []) or []  # 참고 URL

    signals = args.get("signals", "")
    print("Signals:", signals)

    # # reference_urls → http(s)만 남기기
    # invalids = [u for u in raw_refs if not _is_http(u)]
    # if invalids:
    #     print(f"[정화] http(s) 아님 → 제거: {invalids}")
    # reference_urls = [u for u in raw_refs if _is_http(u)]
    reference_urls = raw_refs
    # indices → chat_image_map에 실제 키가 있는 경우만 유지
    indices = []
    if raw_indices:
        i0 = raw_indices[0]
        if isinstance(i0, int) and (i0 in chat_image_map):
            indices = [i0]
        else:
            print(f"[정화] 유효하지 않은 indices → 무시: {raw_indices}")

    # 선택 결과
    generation_prompt = args.get("generate_instructions") or prompt  # generate 프롬프트
    edit_instructions = args.get("edit_instructions")  # edit 프롬프트

    style_transfer = bool(args.get("style_transfer", False))
    image_description = args.get("image_description", "")

    # 편집 계열인데 base 후보가 없을 때 업로드 첫 http URL로 폴백  #
    if subtype == "edit" and not indices and not reference_urls:
        raise ValueError("[에러] 편집 계열인데 base 후보가 없음 (uploads에 http(s) URL 없음)")

    print(f"[분류] action=image task(고정), subtype={subtype}, needs={needs}, style_transfer={style_transfer}")
    if indices:
        print(f"[대상 indices(chat#i)] {indices}")
    if reference_urls:
        print(f"[참조 URL] {reference_urls}")

    # ── 8) 이미지 작업 실행
    payload = {
        "prompt": prompt,
        "subtype": (subtype or "generate"),
        "indices": indices,  # chat 이미지 인덱스
        "reference_urls": reference_urls,  # chat/업로드/기타 모두 file://
        "generate_instructions": (generation_prompt if subtype == "generate" else None),
        "edit_instructions": (edit_instructions if (subtype != "generate") else None),
        "chat_image_map": chat_image_map,
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
        print("[📥] 작업 수신:", body.decode("utf-8"))
        task = json.loads(body)

        # request id
        request_id = task['requestId']
        # 큐 입력 JSON 구조 파싱
        prompt = task.get("prompt", "")
        images_path = task.get("imagesPath", [])
        style_image_id = task.get("styleImageId", "")
        style_image_path = task.get("styleImagePath", "")
        recent_chat = task.get("chat", [])
        chat_summary = task.get("chatSummary", "")

        success, message = classify_and_execute(
            prompt,
            images_path,
            style_image_id,
            style_image_path,
            recent_chat,
            chat_summary)
        print(f"[DEBUG] prompt={prompt}")
        print(f"[DEBUG] images_path={images_path}")
        print(f"[DEBUG] origin_image_id={style_image_id}, style_image_path={style_image_path}")
        print(f"[DEBUG] recent chat count={len(recent_chat)}")
        print(f"[DEBUG] chat_summary={chat_summary}")

        if success == "ok":
            print(message)
            print("Message 수신 성공")
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
            channel.basic_publish(exchange='', routing_key=IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, body=json.dumps(message))
            channel.basic_ack(delivery_tag=method.delivery_tag)

        elif success == "clarify":
            message = {
                "isSuccess": True,
                "requestId": request_id,
                "isImageGenerated": False,
                "textContext": message["reason"],
                "chatSummary": message["chat_summary"],
            }
            channel.basic_publish(exchange='', routing_key=IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, body=json.dumps(message))
            channel.basic_ack(delivery_tag=method.delivery_tag)

        elif success == "error":
            message = {
                "requestId": request_id,
                "isSuccess": False,
                "chatSummary": chat_summary,
                "error": str(message)
            }
            print(f"에러 발생: {message}")
            channel.basic_publish(exchange='', routing_key=IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, body=json.dumps(message))
            channel.basic_ack(delivery_tag=method.delivery_tag)
        else:
            print(f"Fatal error detected: You need to check the error message code({success}) in classify_and_execute!")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as e:
        print("[❌] on_message 에러:", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    context = ssl.create_default_context()
    credentials = pika.PlainCredentials(IMAGE_GENERATION_CHAT_USERNAME, IMAGE_GENERATION_CHAT_PASSWORD)
    params = pika.ConnectionParameters(
        host=IMAGE_GENERATION_CHAT_HOST,
        port=int(IMAGE_GENERATION_CHAT_PORT),
        credentials=credentials,
        ssl_options=pika.SSLOptions(context)
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    dlx_args = {
        'x-dead-letter-exchange': 'ai.image.request.dlx',
        'x-dead-letter-routing-key': 'ai.image.request.retry'
    }
    channel.queue_declare(queue=IMAGE_GENERATION_CHAT_QUEUE, durable=True, arguments=dlx_args)
    channel.queue_declare(queue=IMAGE_GENERATION_CHAT_RESPONSE_QUEUE, durable=True)

    channel.basic_consume(queue=IMAGE_GENERATION_CHAT_QUEUE, on_message_callback=on_message)
    print("[🚀] 작업 대기 중...")
    channel.start_consuming()

# def main():
#     json_path = "./consumer_test.json"
#     raw = False
#     with open(json_path, "r", encoding="utf-8") as f:
#         raw = f.read()
#     if raw:
#         # print(raw)
#         on_message(raw.encode("utf-8"))
#     else:
#         print("reading json error")


if __name__ == '__main__':
    client = get_client()
    main()
