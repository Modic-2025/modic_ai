def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. prompt와 최신 chat을 기반으로 subtype·base·references를 결정한다.
우선 prompt > chat을 우선한다.

[입력 구조 및 참조 규칙]
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

[작업 타입 규칙]
R : 설명 요청 → needs_clarification
R0: 보류/정지 → needs_clarification
R1: 업로드만 있고 prompt 비어있음 → style_transfer=true
R2: 업로드 없음 + 스타일만 요청 → style_transfer=true
R3: 스타일 + 편집 키워드 → edit, style_transfer=true
R4: 편집 키워드(교체/삽입/제거/보정 등) → edit
R5: 입력 이미지 없음 + 생성 요청 → generate
R100: 기타 → 맥락대로 결정
적용순서: R → R0 → 특수 → R1 → R2 → R3 → R4 → R5 → R100

[공통 예외]
요청이 모호하면 needs_clarification=true
사용자의 프롬프트가 일반적인 문장 구조를 따르지 않거나 이해 불가능한 경우
예: "123!!!@@?", "어제 그거 그거 있잖아 그걸로"
모델이 현재 맥락으로 작업 방향을 명확히 결정할 수 없는 경우
→ 위 조건 중 하나라도 해당하면, needs_clarification=true, reason=요청이 모호한 이유를 사용자에게 친절하고 자세하게 정리해 처리한다. 시스템적인 설명이 아니라 설명이 더 필요한 부분을 알려준다.

[특수 예외 – 업로드 + “귀여운/밝은 느낌으로 생성해줘”]
업로드 있고 prompt가 “더/좀 더 + 형용사 + 느낌으로 이미지 생성/만들어줘” 형태면:
- subtype=edit
- style_transfer=false
- base=uploads[0], references=[]
- needs_clarification=true
- edit_instructions에 형용사 포함

[이미지 선택 규칙]
- base: 항상 1개.
- 기본 규칙: 작업에 특별한 지시가 없으면 base=uploads[0], 업로드가 없으면 chat 최신 이미지.
- 사용자가 한 장만 지칭하면 base만 사용하고 references=[]

- **삽입/합성 패턴 (이번 테스트 케이스 포함)**
  - prompt에 "중간에 이 이미지를 넣어줘", "이 이미지를 넣어줘", "이 이미지를 가운데에 넣어줘",
    "배경에 이 이미지를 올려줘" 처럼
    **"이 이미지를 ~에 넣어줘/올려줘/끼워줘"** 형태가 등장하고,
    chat에 AI가 방금 생성한 이미지가 1장 이상 있고(예: 스타일 변환 결과),
    이번 요청에서 uploads로 새 이미지가 들어온 경우:
      - base = chat에서 가장 최신 AI 이미지 (이전까지 만들어진 배경/결과물)
      - references = 업로드된 이미지들 (이번에 "이 이미지"라고 지칭한 것들)
      - 즉, 이전에 생성된 이미지를 배경(base)으로 두고, 업로드 이미지를 그 안에 삽입하는 편집으로 해석한다.

- “방금/최근 만든 이미지” 지칭 시 chat 최신 AI이미지를 base로
- 여러 장 역할 언급 시 base=주어·배경·결과물, references=나머지
- R2(스타일 전용)는 references=[] 유지

[출력 작성 규칙]
- subtype ∈ {generate, edit, style_transfer}
- base, references는 규칙대로 index·path 포함
- generate_instructions / edit_instructions는 subtype에 맞게 간단히 작성
- image_description은 핵심 객체명(예: 고양이/호랑이/고흐)을 반드시 포함
- signals: 판단에 사용된 핵심 키워드
- chat_summary: 한 줄 요약
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
                "reason": {"type": "string", "description": "설명이 필요한 이유를 친절하고 자세하게 정리해 처리한다."},
                "signals": {"type": "array", "items": {"type": "string"}},
                "chat_summary": {"type": "string"}
            },
            "required": ["subtype", "needs_clarification"]
        }
    }
}]