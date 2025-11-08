import argparse
import glob
import json
import os
import sys
import time
import random
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from static.classifier_preprompt import SYSTEM_INSTRUCTIONS, TOOLS
# -------------------------------
# Utilities
# -------------------------------
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)

def _safe(s: Optional[str]) -> str:
    return (s or "").replace("\n", " ").strip()

def _bool(v: Any) -> bool:
    if v is True or v is False:
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes", "y"): return True
        if t in ("false", "0", "no", "n"): return False
    return False

def _build_chat_images(recent_chat: List[Dict]) -> List[Dict]:
    chat_images = []
    i = 0
    for turn in list(recent_chat or []):
        role = turn.get("role", "user")
        for c in turn.get("contents", []):
            if (c.get("type") or "").lower() == "image":
                path = _safe(c.get("imagePath", "")) or ""
                if not path:
                    continue
                chat_images.append({
                    "i": i,
                    "path": path,
                    "role": role,
                    "desc": _safe(c.get("description", "")),
                    "fromOriginImage": _bool(c.get("fromOriginImage"))
                })
                i += 1
    return chat_images

def _resolve_item(item: Optional[Dict], chat_images: List[Dict], uploads: List[str]) -> Optional[str]:
    if not item or "source" not in item:
        return None
    src = item["source"]
    idx = item.get("index")
    pth = item.get("path")
    if isinstance(pth, str) and pth:
        return _safe(pth)
    if src == "chat" and isinstance(idx, int):
        for ci in chat_images:
            if ci["i"] == idx:
                return ci["path"]
    if src == "upload" and isinstance(idx, int):
        if 0 <= idx < len(uploads):
            return _safe(uploads[idx])
    return None

def _normalize_args(args: Dict, chat_images: List[Dict], uploads: List[str]) -> Dict:
    out = {
        "subtype": args.get("subtype"),
        "style_transfer": bool(args.get("style_transfer", False)),
        "needs_clarification": bool(args.get("needs_clarification", False)),
        "edit_instructions": args.get("edit_instructions"),
        "generate_instructions": args.get("generate_instructions"),
        "chat_summary": args.get("chat_summary", ""),
        "image_description": args.get("image_description", ""),
        "reason": args.get("reason", ""),
        "base_path": None,
        "ref_paths": []
    }
    base = args.get("base")
    out["base_path"] = _resolve_item(base, chat_images, uploads)
    refs = args.get("references") or []
    for r in refs:
        rp = _resolve_item(r, chat_images, uploads)
        if rp: out["ref_paths"].append(rp)
    return out

def _normalize_expected(exp: Dict, chat_images: List[Dict], uploads: List[str]) -> Dict:
    out = {
        "subtype": exp.get("subtype"),
        "style_transfer": bool(exp.get("style_transfer", False)),
        "needs_clarification": bool(exp.get("needs_clarification", False)),
        "edit_instructions_contains": exp.get("edit_instructions_contains"),
        "generate_instructions_contains": exp.get("generate_instructions_contains"),
        "base_path": None,
        "ref_paths": []
    }
    out["base_path"] = _resolve_item(exp.get("base"), chat_images, uploads) or _safe((exp.get("base") or {}).get("path"))
    for r in exp.get("references", []) or []:
        out["ref_paths"].append(_resolve_item(r, chat_images, uploads) or _safe(r.get("path")))
    out["ref_paths"] = [p for p in out["ref_paths"] if p]
    return out

def _nullish(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "*"):
        return True
    return False

def _is_specified(d: Dict, key: str) -> bool:
    return (key in d) and (not _nullish(d.get(key)))

# -------------------------------
# 모델 호출 (분류만)
# -------------------------------
def classify_only(client: OpenAI, *, model: str, prompt: str, uploads: List[str],
                  recent_chat: List[Dict], chat_summary: str) -> Tuple[str, Dict]:
    chat_images = _build_chat_images(recent_chat)

    content = [
        {"type": "text", "text": "아래 JSON을 읽고 route_scenario로 분류 결과만 반환하세요."},
        {"type": "text", "text": json.dumps({"type": "prompt", "value": _safe(prompt)}, ensure_ascii=False)},
        {"type": "text", "text": json.dumps({"type": "chat_images", "value": chat_images}, ensure_ascii=False)},
        {"type": "text", "text": json.dumps({"type": "uploads", "value": list(map(_safe, uploads or []))}, ensure_ascii=False)},
    ]
    if chat_summary:
        content.append({"type": "text", "text": json.dumps({"type": "chat_summary", "value": _safe(chat_summary)}, ensure_ascii=False)})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": content},
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "route_scenario"}},
        temperature=0,
        top_p=1,
    )
    choice = resp.choices[0]
    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        return "error", {"error": "툴 호출 없음", "raw": choice.message.content}

    raw = tool_calls[0].function.arguments
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return "error", {"error": "arguments JSON 파싱 실패", "raw": raw}

    args["chat_summary"] = args.get("chat_summary", chat_summary)
    return "ok", {"args": args, "chat_images": chat_images}

# -------------------------------
# 검증
# -------------------------------
def assert_case(actual_args: Dict, chat_images: List[Dict], uploads: List[str], expected: Dict) -> Tuple[bool, str, Dict]:
    actual = _normalize_args(actual_args, chat_images, uploads)
    exp = _normalize_expected(expected, chat_images, uploads)

    diffs = []

    def _cmp(label, a, e):
        if a != e:
            diffs.append(f"- {label}: expected={e!r} actual={a!r}")

    def _cmp_path(label, a, e):
        # 경로 비교: None/""/"null"/"*" 은 동치 처리
        if _nullish(a) and _nullish(e):
            return
        if a != e:
            diffs.append(f"- {label}: expected={e!r} actual={a!r}")

    # 무엇을 비교할지: expected에 '명시적으로' 들어온 것만
    check_subtype = _is_specified(expected, "subtype")
    check_base    = ("base" in expected) or ("base_path" in expected)
    check_refs    = ("references" in expected) or ("ref_paths" in expected)
    check_style   = ("style_transfer" in expected)
    check_clarify = ("needs_clarification" in expected)
    check_edit_contains     = _is_specified(expected, "edit_instructions_contains")
    check_generate_contains = _is_specified(expected, "generate_instructions_contains")

    # subtype 비교 (리스트/와일드카드/널 허용)
    if check_subtype:
        exp_sub = expected["subtype"]
        if isinstance(exp_sub, list):
            if actual["subtype"] not in exp_sub:
                diffs.append(f"- subtype: expected one of {exp_sub!r} actual={actual['subtype']!r}")
        elif exp_sub != "*":
            _cmp("subtype", actual["subtype"], exp["subtype"])

    # base / references 경로 비교: expected에 명시된 경우에만
    if check_base:
        _cmp_path("base_path", actual["base_path"], exp.get("base_path"))

    if check_refs:
        _cmp("ref_paths", actual["ref_paths"], exp.get("ref_paths", []))

    # 플래그류도 명시된 경우에만
    if check_style:
        _cmp("style_transfer", actual["style_transfer"], expected["style_transfer"])

    if check_clarify:
        _cmp("needs_clarification", actual["needs_clarification"], expected["needs_clarification"])

    # 선택 키워드 포함 검사: 명시된 경우에만
    if check_edit_contains:
        key = exp["edit_instructions_contains"] or ""
        has = (actual.get("edit_instructions") or "")
        if key and key not in has:
            diffs.append(f"- edit_instructions_contains: '{key}' not in actual.")

    if check_generate_contains:
        key = exp["generate_instructions_contains"] or ""
        has = (actual.get("generate_instructions") or "")
        if key and key not in has:
            diffs.append(f"- generate_instructions_contains: '{key}' not in actual.")

    passed = len(diffs) == 0
    details = "PASS" if passed else "FAIL:\n" + "\n".join(diffs)
    return passed, details, actual

# -------------------------------
# 케이스 로더 (여러 경로 + 재귀)
# -------------------------------
def _iter_json_files(paths: List[str], recursive: bool) -> List[str]:
    found = []
    patterns = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    for fn in files:
                        if fn.lower().endswith(".json"):
                            found.append(os.path.join(root, fn))
            else:
                patterns.append(os.path.join(p, "*.json"))
        else:
            patterns.append(p)

    for pat in patterns:
        for f in glob.glob(pat):
            if os.path.isdir(f):
                continue
            if f.lower().endswith(".json"):
                found.append(f)

    return sorted(list(dict.fromkeys(found)))

def load_cases(paths: List[str], recursive: bool) -> List[Dict]:
    files = _iter_json_files(paths, recursive)
    cases: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[경고] {f} 파싱 실패: {e}")
            continue
        if isinstance(data, list):
            cases.extend(data)
        else:
            cases.append(data)
    return cases

# -------------------------------
# Runner
# -------------------------------
def run_cases(args) -> int:
    client = get_client()
    cases = load_cases(args.cases, args.recursive)

    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(cases)

    if args.max_cases:
        cases = cases[: args.max_cases]

    total = len(cases)
    if total == 0:
        print("[정보] 실행할 케이스가 없습니다.")
        return 0

    passed = 0
    failed = 0
    last_actual_dump = None

    started = time.time()
    for idx, case in enumerate(cases, 1):
        name = case.get("name", f"case_{idx}")
        prompt = case.get("prompt", "")
        uploads = case.get("uploads", []) or []
        chat = case.get("chat", []) or []
        chat_summary = case.get("chatSummary", "") or ""
        expected = case.get("expected", {}) or {}

        status, payload = classify_only(
            client, model=args.model, prompt=prompt, uploads=uploads,
            recent_chat=chat, chat_summary=chat_summary
        )
        if status != "ok":
            failed += 1
            print(f"[{idx}/{total}] {name}: ERROR - {payload.get('error')}")
            if args.fail_fast:
                break
            continue

        actual_args = payload["args"]
        chat_images = payload["chat_images"]
        ok, details, actual_norm = assert_case(actual_args, chat_images, uploads, expected)
        last_actual_dump = {"name": name, "file": case.get("_file", ""), "actual_args": actual_args, "normalized": actual_norm}

        tag = "PASS" if ok else "FAIL"
        print(f"[{idx}/{total}] {name}: {tag}")
        if not ok:
            print(details)
            print(actual_args)
            failed += 1
            if args.fail_fast:
                break
        else:
            passed += 1

    dur = time.time() - started
    print(f"\n=== SUMMARY ===")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}  Time: {dur:.2f}s")

    if args.dump_json and last_actual_dump:
        try:
            with open(args.dump_json, "w", encoding="utf-8") as fp:
                json.dump(last_actual_dump, fp, ensure_ascii=False, indent=2)
            print(f"[dump] 마지막 응답 저장: {args.dump_json}")
        except Exception as e:
            print(f"[경고] dump 저장 실패: {e}")

    return 0 if failed == 0 else 1

def parse_args():
    p = argparse.ArgumentParser(description="분류 전용 테스트 러너")
    p.add_argument("--cases", nargs="+", required=True,
                   help="케이스 JSON 파일·디렉토리·글롭 패턴(여러 개 가능)")
    p.add_argument("--recursive", action="store_true",
                   help="디렉토리 입력 시 하위 폴더까지 *.json 재귀 탐색")
    p.add_argument("--model", default=os.getenv("CHAT_MODEL", "gpt-4.1"),
                   help="OpenAI chat model")
    p.add_argument("--fail_fast", action="store_true",
                   help="첫 실패에서 중단")
    p.add_argument("--max_cases", type=int, default=0,
                   help="최대 실행 케이스 수")
    p.add_argument("--shuffle", action="store_true",
                   help="케이스 순서 무작위")
    p.add_argument("--seed", type=int, default=42,
                   help="--shuffle 시 랜덤 시드")
    p.add_argument("--dump_json", default="",
                   help="마지막 응답 args를 저장할 경로")
    return p.parse_args()

if __name__ == "__main__":
    sys.exit(run_cases(parse_args()))
