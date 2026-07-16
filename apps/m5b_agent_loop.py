# ST4 — 맨손 에이전트 루프 (Lab 1): 챗봇 UI에 '진짜 도구 쓰는 에이전트'를 얹는다
# [왜] 1절의 m5_chatbot은 챗 UI 4요소를 fake_llm_stream(가짜)으로 익혔다. 그 코드 주석에 "여기에 실제
#      chat.completions.create를 넣으면 실 LLM 연결"이라 적혀 있었다 — 여기서 그 자리에
#      tool-calling 루프를 '직접 손으로' 짜 넣는다. 프레임워크가 숨기는 판단→행동→관찰을 눈으로 보는 것이 목적.
# [흐름] 17강 ReAct 루프(get_weather/calculate)의 'tool-calling 버전' + 1절 챗 UI + 20강 provider 규약.
# [규약] OpenAI 호환 provider만(local/openrouter/openai). base_url만 전환. Anthropic/Claude 금지(20강 동일).
# [비스트리밍] Lab 1의 목적(think-act-observe 직접 구현)은 스트리밍 없이도 100% 달성된다 — 응답을 한 번에
#      받아 tool_calls 유무만 확인하면 된다. 스트리밍(토큰 조각을 이어붙이는 방식)은 뒤쪽 심화(선택)에서 다룬다.
# 실행: python3.11 -m streamlit run apps/m5b_agent_loop.py

import json
import os
import re

import streamlit as st

from apps.m4_sentiment import analyze as _analyze_sentiment  # ST3에서 만든 감성분석 모델을 '도구'로 재사용

# ── provider 규약 (20강 계승, base_url만 전환) ──
PROVIDERS = {
    "local": {"base_url": "http://localhost:11434/v1", "model": "hermes3:8b", "key": "ollama"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free", "key_env": "OPENROUTER_API_KEY"},
    "openai": {"base_url": None, "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
}

# ── ① 도구 2개: 정의(스키마) + 구현(함수) ──
# [왜] LLM은 도구를 실행하지 않는다 — JSON 스키마(메뉴판)를 보고 "이걸 쓰겠다"고 결정만 한다(17강).
#      description은 사람용 주석이 아니라 LLM이 실제로 읽고 도구를 고르는 유일한 힌트다.
TOOLS = [
    {"type": "function", "function": {
        "name": "analyze_sentiment",
        "description": "한국어 문장의 긍정/부정 감성을 분석한다.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string", "description": "분석할 한국어 문장"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "산술 수식을 계산한다.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "계산할 수식 (예: 150 * 8500)"}},
                       "required": ["expression"]}}},
]


def _tool_analyze_sentiment(text: str) -> str:
    out = _analyze_sentiment(text)  # ST3 KoELECTRA 모델
    return f"감성: {out['label']} (확신도 {out['score']:.1%})"


def _tool_calculate(expression: str) -> str:
    # [보안] eval은 임의 코드 실행 위험 — 허용 문자만 통과(17강의 안전 계산 계승).
    if all(c in set("0123456789+-*/.() ") for c in expression):
        try:
            return f"{expression} = {eval(expression)}"
        except Exception as e:  # noqa: BLE001
            return f"계산 오류: {e}"
    return "허용되지 않는 수식입니다."


TOOL_FUNCS = {"analyze_sentiment": _tool_analyze_sentiment, "calculate": _tool_calculate}


def run_tool(name: str, args: dict) -> str:
    """이름으로 실제 함수를 찾아 실행(4강 딕셔너리 디스패치). 예외는 문자열로 돌려 루프가 멈추지 않게."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"알 수 없는 도구: {name}"
    try:
        return fn(**args)
    except Exception as e:  # noqa: BLE001
        return f"도구 '{name}' 오류: {e}"


# ── provider 준비 (키 로딩·가용성·클라이언트) ──
def _get_key(provider: str) -> str:
    cfg = PROVIDERS[provider]
    if "key" in cfg:
        return cfg["key"]
    return os.environ.get(cfg["key_env"], "")


def provider_available(provider: str) -> bool:
    if provider == "local":
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
            return True
        except Exception:
            return False
    return bool(_get_key(provider))


@st.cache_resource
def get_client(provider: str):
    from openai import OpenAI
    cfg = PROVIDERS[provider]
    return OpenAI(base_url=cfg["base_url"], api_key=_get_key(provider) or "none")


# ── ② 맨손 think-act-observe 루프 (Lab 1의 심장, 비스트리밍) ──
def run_agent(question: str, provider: str, max_turns: int = 5):
    """[Lab 1] 프레임워크 없이 직접 짠 tool-calling 루프 — 비스트리밍(응답을 한 번에 받는다).
    판단(LLM 전체 응답 수신) → 행동(run_tool 실행) → 관찰(결과를 messages에 append 후 재요청)을 반복한다.
    키가 없으면 키워드 라우터(_fake_route)로 폴백해 배포 URL이 키 없이도 동작한다.
    반환값: (최종 답변 문자열, [(도구이름, 결과문자열), ...] 호출 로그) — 로그는 st.status 표시·시각화에 쓴다."""
    if not provider_available(provider):
        return _fake_route(question)
    try:
        client = get_client(provider)
        model = PROVIDERS[provider]["model"]
    except Exception as e:  # noqa: BLE001 — openai 미설치·SDK 초기화 실패도 크래시 대신 안내
        return f"⚠️ LLM 클라이언트 초기화 실패: {e}", []

    messages = [
        {"role": "system", "content": "너는 감성분석·계산 도구를 쓸 수 있는 한국어 조수다. 필요하면 도구를 호출하고 그 결과로 답해라."},
        {"role": "user", "content": question},
    ]
    trace = []  # [호출 로그] (도구이름, 결과) 튜플 — st.status 표시 + 이번 절 인라인 시각화에 재사용
    for _ in range(max_turns):
        # [판단] ① messages+tools를 LLM에 보낸다 — 이번 턴에 tool을 쓸지 최종 답을 낼지는 LLM이 정한다.
        # [안전] LLM 호출을 try/except로 감싼다 — 만료 키·rate limit·네트워크 순단 시 raw traceback
        #        대신 안내 메시지로 안전 종료(위 브릿지의 requests 예외처리와 같은 원칙).
        try:
            response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        except Exception as e:  # noqa: BLE001
            return f"⚠️ LLM 호출 실패: {e}", trace

        msg = response.choices[0].message
        tool_calls = msg.tool_calls  # [비스트리밍] 응답이 한 번에 다 온다 — 조각을 이어붙일 필요가 없다

        # [판단 결과] tool_calls가 비어 있다 = "도구가 더 필요 없다"는 LLM의 판단 → 최종 답변으로 루프 종료.
        if not tool_calls:
            return msg.content or "", trace  # 도구가 더 필요 없는 턴 = 최종 답변

        # [행동 준비] LLM이 고른 tool_calls를 대화 기록에 먼저 남긴다.
        #           [흐름] 이 append가 아래 tool 결과 append보다 반드시 먼저 와야 한다(위 흔한 실수 참고).
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        # [행동] ② LLM이 요청한 도구를 실제로 실행한다 — LLM은 "부르겠다"고 선언만 했을 뿐, 실행은 우리 코드가 한다.
        for tc in tool_calls:
            # [안전] LLM이 깨진 JSON 인자를 보낼 수 있다(17강이 경고한 스키마 이탈 리스크) — json.loads도 방어한다.
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(tc.function.name, args)  # [흐름] 실제 함수 실행(딕셔너리 디스패치, 4강)
            trace.append((tc.function.name, result))
            # [관찰] ③ 실행 결과를 tool 메시지로 messages에 되돌려 넣는다.
            #        [흐름] 다음 for _ in range(max_turns) 반복에서 LLM이 이 결과를 보고
            #        "최종 답을 낼지, 도구를 하나 더 쓸지"를 다시 판단한다 — 이게 반복 구조의 핵심이다.
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # [안전] max_turns 소진 = 도구만 반복하다 최종 답에 도달 못함. 조용히 끝내지 않고 안내(17강 계승).
    return "⚠️ 최대 턴 수에 도달해 종료합니다. 질문을 더 단순하게 나눠보세요.", trace


def _fake_route(question: str):
    """키 없이 동작하는 데모 — 키워드로 도구를 고른다(실 LLM은 같은 도구를 스스로 고른다).
    반환값 형식은 run_agent와 동일: (답변 문자열, 호출 로그)."""
    prefix = "🧪 **데모 모드**(LLM 키 없음) — 키워드로 도구를 고릅니다. 키를 넣으면 LLM이 스스로 고릅니다.\n\n"
    if any(k in question for k in ["감성", "감정", "리뷰", "기분"]):
        text = question.split(":", 1)[1].strip() if ":" in question else question
        result = run_tool("analyze_sentiment", {"text": text})
        return prefix + result, [("analyze_sentiment", result)]
    if any(c.isdigit() for c in question):
        expr = "".join(re.findall(r"[0-9+\-*/.() ]", question)).strip()
        result = run_tool("calculate", {"expression": expr})
        return prefix + result, [("calculate", result)]
    return prefix + "감성분석 또는 계산 질문을 해보세요. 예) `감성분석: 이 영화 최고예요` / `150 * 8500`", []


# ── ③ 1절 챗 UI 4요소(chat_message·chat_input·session_state·status) 재사용 ──
def main():
    st.set_page_config(page_title="맨손 에이전트 루프", page_icon="🤖", layout="centered")
    st.title("🤖 맨손 에이전트 루프 (Lab 1)")
    st.caption("1절의 챗 UI에 진짜 tool-calling 루프를 얹었다 — 비스트리밍 · 키 없으면 데모 모드로 동작")

    with st.expander("💡 이전 챕터와 연결"):
        st.markdown(
            "- **17강 ReAct 루프**: `get_weather`/`calculate` 도구로 CLI에서 짰던 think-act-observe를 "
            "그대로 Streamlit 챗 UI 안에 옮겼습니다.\n"
            "- **ST3 감성분석 모델**: `analyze_sentiment` 도구는 ST3에서 만든 KoELECTRA 모델(`apps/m4_sentiment.py`)을 그대로 재사용합니다."
        )

    provider_names = list(PROVIDERS)
    provider = st.sidebar.selectbox("Provider", provider_names, index=provider_names.index("openai"))
    if not provider_available(provider):
        st.info("🧪 데모 모드 — provider 키/서버가 없어 키워드 라우터로 동작합니다. 키를 넣으면 LLM이 도구를 스스로 고릅니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("예) 감성분석: 이 영화 최고예요  /  150 * 8500"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("에이전트가 판단하는 중...", expanded=True) as status:
                answer, trace = run_agent(prompt, provider)
                for name, result in trace:
                    st.write(f"🔧 `{name}` 호출 → {result}")
                status.update(label="완료", state="complete", expanded=False)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
