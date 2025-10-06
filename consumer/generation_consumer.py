# import osssl
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

from typing import List, Dict, Any, Optional
from openai import OpenAI

# Load env variable & Model
from static.rabbitmq import *
from static.model import *
from static.s3 import *


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
    for i, ref in enumerate(ref_list):
        if not ref:
            continue
        try:
            rn, rf, rct = open_binary(ref)
            # 서버가 인식하면 활용, 무시해도 안전
            files[f"ref_image_{i}"] = (rn, rf, rct)
        except Exception as e:
            print(f"[경고] reference 이미지 로드 실패({ref}): {e}")

    # 선택: style image
    if style_image_path:
        try:
            sn, sf, sct = open_binary(style_image_path)
            files["style_image"] = (sn, sf, sct)
        except Exception as e:
            print(f"[경고] style 이미지 로드 실패({style_image_path}): {e}")

    # 참고/스타일 안내를 프롬프트에 주입
    ref_hint = ""
    if ref_list:
        ref_hint += f" 참고이미지 {len([r for r in ref_list if r])}장을 반영해 편집하라."
    if style_image_path:
        ref_hint += " style_image의 화풍/질감/톤을 참고하라."
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
        try:
            print("[OpenAI error payload]", resp.status_code, resp.text)
        except Exception:
            pass
    resp.raise_for_status()
    # 파일 핸들러 정리 (requests가 닫지만 안전하게)
    try:
        base_fh.close()
        if mask_path:
            mask_fh.close()
        for k, v in files.items():
            if k in ("image", "mask"):
                continue
            # v는 (name, fh, ctype)
            try:
                v[1].close()
            except Exception:
                pass
    except Exception:
        pass

    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    return Image.open(BytesIO(base64.b64decode(b64)))


def do_style_transfer(style_image_path):
    print("[Style transfer]: not yet. style image path: style_image_path")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 1) 시스템 지침 (베이스/참고 이미지 선택 및 edit 프롬프트 생성 규칙 명시)
# ──────────────────────────────────────────────────────────────────────────────
def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. 다음 규칙을 반드시 따른다.

[작업 타입 결정]
- 이미지가 하나라도 주어지고 수정 요청이 있으면 subtype=edit (또는 recolor_object).
- generate는 오직 입력 이미지가 전혀 없거나, 사용자가 새로운 지시로 생성을 요청한 경우.

[베이스 선택 우선순위]
1) uploads(images_path)가 비어있지 않다면: reference_urls[0] = uploads[0], indices는 비워둔다.
2) uploads가 비어있고 chat에 이미지가 있다면: indices[0] = 해당 chat 이미지 인덱스(정수). -1 사용 금지.

[참고 이미지]
- 추가 참고가 필요하면 reference_urls에 뒤에 이어서 넣는다(HTTP(S) URL 또는 S3 Key 그대로, 검증/변환 금지).

[프롬프트 작성]
- edit/recolor/style: 사용자의 요청을 구체화하여 edit_instructions에 작성.
- 배경 교체 등 부분 편집일 때는 “피사체/전경/얼굴/손/의상/소지품은 유지, 해당 부분(배경 등)만 변경”을 명시적으로 포함.
- style transfer 의도가 분명할 때만 style_transfer=true.

[clarify]
- uploads도 chat 이미지도 없고, 요청 의도도 불명확할 때만 needs_clarification=true.

[출력 형식]
- subtype, edit_instructions, indices, reference_urls, target_objects, target_colors, style_transfer, needs_clarification, reason, chat_summary를 반환.
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
            "요약본(컨텍스트), 시스템 규칙, 채팅 요약, 현재 입력을 바탕으로 "
            "세부 편집 요구에 맞게 "
            "'indices[0]'에 편집 베이스를, 'reference_urls'에는 참고 이미지 **http(s) URL**을 채우고, "
            "'response', 'generate_instructions', edit/recolor/style은 'edit_instructions'를 작성한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtype": {
                    "type": "string",
                    "enum": ["generate", "edit", "recolor_object"],
                    "description": "이미지 작업 세부 타입(스타일 변환은 style_transfer=true), 무조건 하나는 지정해야 됨."
                },
                "reference_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "참고 이미지 목록. **http(s) URL 또는 S3 Key** 그대로 넣기(검증/변환 금지)."
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "chat 이미지 선택 시: indices[0] = chat#i의 i (정수). **-1 사용 금지**."
                },
                "generate_instructions": {"type": "string", "description": "이미지 '생성' 프롬프트(구체적으로)"},
                "edit_instructions": {"type": "string", "description": "최대한 사용자의 prompt에 맞춰 편집/채색/스타일 변환 지시문(구체적으로)"},

                "target_objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "색 변경/편집 대상 오브젝트(recolor에서 권장)"
                },
                "target_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "오브젝트별 목표 색(recolor에서 권장)"
                },

                "image_description": {
                    "type": "string",
                    "description": "생성할 이미지에 대한 설명을 반환합니다. 이 설명은 나중에 이미지에 대해서 참고할 때 쓰입니다."
                },

                "style_transfer": {
                    "type": "boolean",
                    "description": "스타일 변환 필요 여부(true면 style transfer)"
                },

                "needs_clarification": {"type": "boolean", "description": "추가 정보 필요 여부"},
                "reason": {"type": "string", "description": "애매한 표현이나 판단 근거 또는 부족 정보에 대해 사용자에게 구체적으로 설명하고 다음 행동 추천."},
                "signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "탐지된 키워드/신호(디버깅용)"
                },
                "chat_summary": {"type": "string", "description": "지금까지의 채팅을 요약한 글,"}
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
    target_objects: List[str],
    target_colors: List[str],
    style_transfer: bool,
    style_image_path: Optional[str] = None,
    chat_image_map: Optional[Dict[int, str]] = None,
) -> (bool, str, object):
    chat_image_map = chat_image_map or {}

    # 출력 경로 준비
    os.makedirs("outputs", exist_ok=True)
    existing = [f for f in os.listdir("outputs") if f.startswith("img_") and f.endswith(".png")]
    out_path = os.path.join("outputs", f"img_{len(existing)+1:03d}.png")

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
            return True, "", img
        except Exception as e:
            print(f"[에러]: {e}")
            return False, f"[에러]: {e}", None

    base_path = None
    if indices:
        cand = chat_image_map.get(indices[0])
        if cand:
            base_path = cand

    if not base_path:
        return False, "[에러] 편집 base 이미지를 찾지 못했습니다.", None

    extra_refs = []
    if reference_urls and len(reference_urls) > 1:
        extra_refs = reference_urls[1:]

    # 편집 지시문
    edit_text = (edit_instructions or "").strip()
    if subtype == "recolor_object" and not edit_text:
        pairs = []
        for i, obj in enumerate(target_objects or []):
            col = target_colors[i] if i < len(target_colors or []) else ""
            if obj and col:
                pairs.append(f"{obj}를 {col} 색으로")
        if pairs:
            edit_text = " / ".join(pairs) + " 바꿔줘."
    if not edit_text:
        edit_text = "이미지를 개선해줘"

    # 스타일 변환 힌트
    if style_transfer and style_image_path:
        do_style_transfer(style_image_path)
        print(f"[정보] style_transfer=True, style_image_path={style_image_path}")

    print(f"[편집] base={base_path}, refs={extra_refs}, instr={edit_text!r}")
    try:
        img = edit_image_from_text(
            image_path=base_path,
            prompt=edit_text,
            size="1024x1024",
            mask_path=None,
            reference_image_paths=extra_refs,
            style_image_path=style_image_path if style_transfer else None,
        )
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
        return isinstance(u, str) and u.startswith(("http://", "https://"))

    # ── 1) content 배열 준비 (텍스트+이미지를 한 메시지에 동시 포함)
    content = []

    # 안내/규칙 텍스트
    content.append({
        "type": "text",
        "text": (
            "아래는 한 번에 제공되는 대화 맥락과 이미지들입니다.\n"
            "- recent chat 섹션: 사용자/어시스턴트가 주고받은 텍스트와 이미지(라벨: chat#i)\n"
            "- uploads 섹션: 사용자가 업로드한 이미지(라벨: upload#j)\n"
            "선택 규칙:\n"
            "1) 편집/상세편집일 경우 base 이미지를 반드시 지정:\n"
            "   - chat 목록에서 고르면: indices[0] = i   (i는 chat#i의 i)\n"
            "2) 참고 이미지는 http(s) URL 혹은 S3 Key임\n"
            "3) edit/recolor/style_transfer면 사용자의 prompt를 바탕으로 edit_instructions 구체적으로 작성(모호 표현 금지)\n"
            "4) generate면 prompt를 구체화하고 indices/reference_urls 비움\n"
        )
    })

    content.append({
        "type": "text",
        "text": f"\n### [C] user prompt\n{_safe(prompt) or '(빈 prompt)'}"
    })

    # ── 2) chat 섹션: 텍스트/이미지를 턴 순서대로 넣되, 이미지에는 chat#index 라벨 부여
    content.append({"type": "text", "text": "\n### [A] recent chat 섹션 (채팅 순서 그대로)\n"})
    chat_image_map: Dict[int, str] = {}
    img_counter = 0

    for turn in (recent_chat or []):
        # role 없으면 user로 표시
        role = turn.get("role", "user")
        # 턴 헤더 (텍스트가 없어도 턴 존재를 표시)
        content.append({"type": "text", "text": f"- [{role}] "})

        # 이 턴의 contents를 순서대로 펼침
        for c in turn.get("contents", []):
            ctype = (c.get("type") or "").lower()
            if ctype == "text":
                text = _safe(c.get("text", ""))
                if text:
                    content.append({"type": "text", "text": f"  • text: {text}"})
            elif ctype == "image":
                img_path = _safe(c.get("imagePath", ""))
                desc = _safe(c.get("description", ""))
                from_origin_image = _bool(c.get("fromOriginImage"))
                if not img_path:
                    continue
                file_url = img_path
                # if img_path.startswith("http"):
                #     file_url = img_path
                # else:
                #     print("[에러] 이미지 URL이 http(s)가 아닙니다. chat.completions에서 무시될 수 있습니다.")
                #     return "error", "[경고] 이미지 URL이 http(s)가 아닙니다. chat.completions에서 무시될 수 있습니다."
                # 이미지 라벨 안내 텍스트 + 메타
                label_text = (f"  • image chat#{img_counter} | desc={desc} | path={img_path} | "
                              f"from_origin_image={from_origin_image}")
                content.append({"type": "text", "text": label_text})
                # 실제 이미지 첨부
                # content.append({"type": "image_url", "image_url": {"url": file_url}})
                chat_image_map[img_counter] = file_url
                img_counter += 1

    if img_counter == 0:
        content.append({"type": "text", "text": "  (chat 섹션에 이미지 없음)"})

    # ── 3) 업로드 풀 섹션
    content.append({"type": "text", "text": "\n### [B] uploads 섹션 (images_path)\n"})
    if images_path:
        for j, p in enumerate(images_path):
            p = _safe(p)
            # if not p:
            #     continue
            # if not p.startswith(("http://", "https://")):
            #     raise ValueError(f"[에러] 업로드 URL이 http(s)가 아님. 무시: {p}")
            content.append({"type": "text", "text": f"- upload#{j} | s3_key={p}"})
    else:
        content.append({"type": "text", "text": "(업로드 풀 비어있음)"})

    # # ── 4) 요약/추가 맥락
    # if chat_summary:
    #     content.append({"type": "text", "text": f"\n### chat_summary\n{_safe(chat_summary)}"})

    print(content)
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

    subtype = args.get("subtype")  # "generate" | "edit" | "recolor_object" | "style_transfer"

    # 선택 결과 (정화(sanitize) 포함)  # ★
    raw_indices = args.get("indices", []) or []  # chat 이미지 선택 시: indices[0] = i
    raw_refs = args.get("reference_urls", []) or []  # 참고 URL

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

    target_objects = args.get("target_objects", []) or []
    target_colors = args.get("target_colors", []) or []
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
        "target_objects": target_objects,
        "target_colors": target_colors,
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
                "chatSummary": chat_summary
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
