def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. 이번 요청과 chat 맥락을 기준으로 subtype과 이에 따른 출력을 결정한다.

1. 입력 구조 및 참조 규칙
chat에는 **최신 N개의 대화(turn)** 가 저장되어 있다. 대화는 TEXT와 IMAGE 2 종류가 있다.
각 대화는 다음 구조를 가진다:
{
  "role": "USER" | "AI",
  "contents": [
    {"type": "TEXT" | "IMAGE",
     "text": <문자열 or null>,
     "imagePath": <이미지 경로 or null>,
     "description": <이미지 설명 or null>,
     "fromOriginImage": <bool>}
  ]
}
- index 0 이 최신 메시지.
- uploads(images_path): 이번 요청 이미지 경로 목록.
- fromOriginImage: 스타일 변환 여부

2. 우선순위
1) 이번 요청이 최우선
2) 이번 요청이 애매할 경우 chat 맥락에 맞게 작업 타입 및 지칭 결정

3. 작업 타입 결정 (R-규칙, 순서 고정)
R 기능 설명 → needs_clarification=true, style_transfer=false, reason=기능 요약
R0 보류/정지 → needs_clarification=true, style_transfer=false, subtype 없음, reason=""
R1 업로드만(prompt 없이 images만) → subtype=style_transfer, style_transfer=true, base=uploads[0], needs_clarification=false
R2 스타일 전용(image 없이 prompt에 스타일 키워드만) → subtype=style_transfer, style_transfer=true, "needs_clarification": false
R3 혼합(스타일+편집 키워드) → subtype=edit, style_transfer=true
R4 편집(교체/삽입/제거/변경/보정 등) → subtype=edit
R5 생성(입력 이미지 없음) → subtype=generate
R100 다른 규칙에 해당하지 않는 사항 → 기능에 맞게 잘 해석해 처리

단 R3~R100은 아래 예외 사항이 발생할 경우 예외 처리
예외 처리 조건:
사용자의 요청이 불분명하거나 의미가 모호한 경우
예: "대충 처리해줘", "알아서 잘 해봐", "그 느낌으로" 등
사용자의 프롬프트가 일반적인 문장 구조를 따르지 않거나 이해 불가능한 경우
예: "123!!!@@?", "어제 그거 그거 있잖아 그걸로"
모델이 현재 맥락으로 작업 방향을 명확히 결정할 수 없는 경우
→ 위 조건 중 하나라도 해당하면, "needs_clarification": true, "reason": "사용자의 요청이 모호하거나 불분명하여 추가적인 설명이 필요함" 으로 처리한다.

적용 순서: R→R0→R1→R2→R3→R4→R5->R100

4. base / references
base: { "source": "chat"|"upload", "index": <int>, "path"?: <str> }
references: [ { "source": ..., "index"?: <int>, "path"?: <str> }, ... ] 
사용자의 요청과 채팅 맥락에 맞게 작업에 필요한 이미지를 chat_images(최근 대화의 이미지 목록, index=0이 최신)와 uploads(이번 요청에 포함된 업로드 이미지, index=0부터 순서대로)에서 골라 base 혹은 references 이미지로 결정한다.
base는 subtype edit, style transfer의 중심이 되는 1개의 이미지로, 별다른 지시가 없을 경우 업로드가 있으면 uploads[0] 없으면 최신 chat_images[0]을 사용한다. 
references는 편집이나 변환 시 참고할 추가 이미지들의 배열이며, base는 절대 포함하지 않는다. references는 중요도 순서대로 나열하며, path를 사용한다.
ex) A를 B에 넣어줘. => base:B, references: A

출력(JSON): subtype, base, references, generate_instructions, edit_instructions,
style_transfer, needs_clarification, reason, chat_summary, signals, image_description
""".strip()

SYSTEM_INSTRUCTIONS = build_system_instructions()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "route_scenario",
        "description": "작업 타입(subtype)을 결정하고, base와 references를 '구조화'하여 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtype": {"type": "string", "enum": ["generate", "edit", "style_transfer"]},
                "base": {
                    "type": "object",
                    "description": "실제로 스타일 변환· 이미지 편집의 중심이 되는 이미지(항상 1개)",
                    "properties": {
                        "source": {"type": "string", "enum": ["chat", "upload"]},
                        "index": {"type": "integer"},
                        "path": {"type": "string"}
                    },
                    "required": ["source"]
                },
                "references": {
                    "type": "array",
                    "description": "편집 또는 변환 시 참고되는 부가 이미지 목록 (0개 이상). base의 이미지는 넣지 않는다.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "enum": ["chat", "upload"]},
                            "index": {"type": "integer"},
                            "path": {"type": "string"}
                        },
                        "required": ["source"]
                    }
                },
                "generate_instructions": {
                    "type": "string",
                    "description": "subtype ∈ {generate}인 경우 사용자의 prompt와 chat 맥락에 맞게 자세히 작성"
                },
                "edit_instructions": {
                    "type": "string",
                    "description": "subtype ∈ {edit}인 경우 사용자의 prompt와 chat 맥락에 맞게 base와 references 순서를 맞추서 맞게 자세히 작성"
                },
                "style_transfer": {"type": "boolean"},
                "image_description": {"type": "string"},
                "needs_clarification": {"type": "boolean"},
                "reason": {"type": "string"},
                "signals": {"type": "array", "items": {"type": "string"}},
                "chat_summary": {"type": "string"}
            },
            "required": ["subtype", "needs_clarification"]
        }
    }
}]