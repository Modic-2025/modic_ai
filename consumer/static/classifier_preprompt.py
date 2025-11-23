def build_system_instructions() -> str:
    return """
너는 '이미지 편집 플래너'다. prompt와 최신 chat을 기반으로 subtype·base·references를 결정한다.
우선 prompt > chat을 우선한다.

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

[특수 예외 – 업로드 + “귀여운/밝은 느낌으로 생성해줘”]
업로드 있고 prompt가  
“더/좀 더 + 형용사 + 느낌으로 이미지 생성/만들어줘” 형태면:
- subtype=edit
- style_transfer=false
- base=uploads[0], references=[]
- needs_clarification=true
- edit_instructions에 형용사 포함

[이미지 선택 규칙]
- base: 항상 1개.
- 기본: uploads[0] → 없으면 chat 최신 이미지.
- 사용자가 한 장만 지칭하면 base만 사용하고 references=[]
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
                "reason": {"type": "string"},
                "signals": {"type": "array", "items": {"type": "string"}},
                "chat_summary": {"type": "string"}
            },
            "required": ["subtype", "needs_clarification"]
        }
    }
}]