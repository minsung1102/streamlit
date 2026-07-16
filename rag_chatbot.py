# ST4 — 나만의 PDF RAG 챗봇: 청킹 → 임베딩 검색 → 컨텍스트 조립 → LLM 답변(+출처 표시)
# [왜] 1절에서 챗 UI 4요소를, 3절에서 도구(calculate)의 스키마-구현 분리를 배웠다. 이 앱은 그 지식을
#      "내 PDF에 대해 답하는 챗봇"에 적용한 완성형이다 — 검색(retrieve, 아래)이 이번 절의 메인 "도구"다.
#      (rag_lite.search_docs()는 본문에서 쓰지 않는 도전 과제용 함수 — retrieve()는 _top_matches()를 직접 쓴다.)
# [흐름] 19강 미니 RAG(sentence-transformers 임베딩)의 "청킹→임베딩→검색→생성" 4단계를 그대로 따른다.
#      다만 "무엇을 검색할지" 판단은 사용자 질문 자체이므로, 이 앱은 매 턴 검색을 직접 한 번 실행한다
#      (행동) → 결과를 LLM에게 context로 전달한다(관찰) → LLM이 그 근거로 답한다(최종 답변).
#      LLM이 검색 여부까지 스스로 판단하는 완전 자동 에이전트 버전은 아래 main()의 🔬 심화 노트 참고.
# [재사용] apps/rag_lite.py: chunk_text·_extract_pdf_text(PDF 추출)·build_index/_top_matches(TF-IDF)·
#      EMBED_PROVIDERS/embed_provider_available/embed_texts/search_docs_embed(임베딩 검색, 이번 절 신규).
#      apps/m5b_agent_loop.py: PROVIDERS·get_client·provider_available(20강 LLM provider 규약, 그대로 재사용).
# [규약] LLM은 OpenAI 호환 provider만(local/openrouter/openai). 임베딩도 동일 3-provider 구조(대칭).
# [신규 코드] 이 파일에서 새로 작성한 부분은 retrieve()의 폴백 분기 + assemble_context() + answer_with_context()
#      뿐이다 — 나머지(청킹·검색·provider 판단·LLM 호출)는 모두 위 두 파일에서 가져와 쓴다.
# 실행: python3.11 -m streamlit run apps/rag_chatbot.py

import streamlit as st

from apps.m5b_agent_loop import PROVIDERS, get_client, provider_available
from apps.rag_lite import (
    EMBED_PROVIDERS,
    _extract_pdf_text,
    _top_matches,
    build_index,
    chunk_text,
    embed_provider_available,
    embed_texts,
    search_docs_embed,
)


def retrieve(query: str, chunks: list[str], embed_provider: str, top_k: int = 3):
    """[검색 — 이번 절의 메인 '도구'] 임베딩 provider를 쓸 수 있으면 의미 기반 검색, 아니면 TF-IDF로 폴백한다.
    반환 형식은 rag_lite의 _top_matches/search_docs_embed와 동일: [(청크 인덱스, 청크 본문, 유사도), ...]."""
    if embed_provider_available(embed_provider):
        try:
            vectors = embed_texts(tuple(chunks), embed_provider)
            return search_docs_embed(query, chunks, vectors, embed_provider, top_k=top_k)
        except Exception as e:  # noqa: BLE001 — 임베딩 호출이 실패해도 검색 자체는 계속돼야 한다
            st.warning(f"⚠️ 임베딩 호출 실패({e}) — TF-IDF로 폴백합니다.")
    # [폴백] 임베딩 provider가 없거나 실패하면 rag_lite의 TF-IDF 검색으로 대체한다 — "임베딩 없이도 구조는 동일".
    vectorizer, matrix = build_index(tuple(chunks))
    return _top_matches(query, chunks, vectorizer, matrix, top_k=top_k)


def assemble_context(matches) -> str:
    """[신규 — 컨텍스트 조립] 검색된 청크를 "[문서 N]" 헤더로 이어붙여 LLM 프롬프트에 넣을 문자열을 만든다.
    N은 청크의 원래 인덱스가 아니라 "이번 질문에 몇 번째로 관련 있었는가"(검색 순위)다."""
    return "\n\n".join(f"[문서 {rank}]\n{chunk}" for rank, (_, chunk, _) in enumerate(matches, start=1))


def answer_with_context(question: str, context: str, provider: str) -> str:
    """[신규] 검색된 컨텍스트를 근거로 LLM에게 답을 요청한다 — 도구 호출 루프 없는 단순 완성 호출이다
    (검색은 이미 위 retrieve()에서 끝났으므로 4절의 tool-calling 루프는 필요 없다).
    키가 없으면 안내 문구만 반환한다 — 검색된 문서 조각 자체는 main()의 '📄 출처 청크 보기' expander가
    이미 따로 보여주므로, 여기서 context를 또 포함하면 화면에 같은 내용이 두 번 노출된다."""
    if not provider_available(provider):
        return "🧪 **데모 모드**(LLM 키 없음) — 아래 '📄 출처 청크 보기'에서 검색된 문서 조각을 확인하세요."
    try:
        client = get_client(provider)
        model = PROVIDERS[provider]["model"]
        messages = [
            {"role": "system", "content": "너는 주어진 문서 조각만 근거로 한국어로 답하는 조수다. 문서에 없는 내용은 모른다고 답해라."},
            {"role": "user", "content": f"[문서]\n{context}\n\n[질문]\n{question}"},
        ]
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        return f"⚠️ LLM 호출 실패: {e}"


def main():
    st.set_page_config(page_title="나만의 PDF RAG 챗봇", page_icon="📚", layout="centered")
    st.title("📚 나만의 PDF RAG 챗봇")
    st.caption("PDF 업로드 → 청킹 → 임베딩 검색 → 컨텍스트 조립 → LLM 답변(+출처 표시)")

    with st.expander("💡 이전 챕터와 연결"):
        st.markdown(
            "- **19강 미니 RAG**: sentence-transformers 임베딩으로 청킹→임베딩→검색→생성 4단계를 배웠습니다. "
            "이 앱은 그 4단계를 그대로 따르되, 임베딩을 **local(Ollama)·openrouter·openai** 3-provider 매트릭스로 "
            "확장합니다(20강 LLM `PROVIDERS`와 대칭 구조).\n"
            "- **3절의 `calculate` 도구**: 정의(스키마)와 구현을 분리하는 원칙이 이 앱의 검색 함수"
            "(`search_docs_embed`/`_top_matches`)에도 그대로 적용됩니다.\n"
            "- **감성분석 도구는 어디로?** 이전 절 `apps/m5b_agent_loop.py`에 이미 등록돼 있습니다 — 오늘은 검색"
            "(`retrieve`)에 집중하기 위해 본문에서는 다루지 않습니다. 다시 붙여보고 싶다면 '도전 과제'를 참고하세요."
        )

    with st.expander("🔬 심화(선택): search_docs를 진짜 '도구'로 등록하면?"):
        st.markdown(
            "아래 `retrieve()`는 매 턴 직접 호출하는 함수입니다. 이걸 4절에서 배운 `TOOLS`/`TOOL_FUNCS` 패턴 그대로 "
            "등록하면(스키마 예: `{\"name\": \"search_docs\", \"parameters\": {\"query\": \"string\"}}`), "
            "LLM이 **검색이 필요한지 스스로 판단**하는 완전한 에이전트가 됩니다 — 실습 'echo_length 도구 추가'와 "
            "정확히 같은 절차(① 스키마 → ② 함수 → ③ `TOOLS.append` → ④ `TOOL_FUNCS` 등록)입니다."
        )

    with st.sidebar:
        st.subheader("⚙️ Provider 설정")
        embed_names = list(EMBED_PROVIDERS)
        embed_provider = st.selectbox("임베딩 provider", embed_names, index=embed_names.index("local"))
        if not embed_provider_available(embed_provider):
            st.caption("🔤 키/서버 없음 → 검색 시 TF-IDF로 자동 폴백합니다.")
        llm_names = list(PROVIDERS)
        llm_provider = st.selectbox("LLM provider", llm_names, index=llm_names.index("openai"))
        if not provider_available(llm_provider):
            st.caption("🧪 키/서버 없음 → 답변 시 데모 모드로 동작합니다.")

    uploaded = st.file_uploader("문서 업로드 (txt·md·pdf)", type=["txt", "md", "pdf"])
    if uploaded is None:
        st.info("문서를 업로드하면 채팅으로 질문할 수 있습니다.")
        return

    if uploaded.name.lower().endswith(".pdf"):
        raw_text = _extract_pdf_text(uploaded.getvalue())
        if not raw_text.strip():
            st.warning("⚠️ 텍스트를 추출하지 못했습니다 — 스캔본(이미지) PDF는 OCR이 별도로 필요합니다.")
            return
    else:
        raw_text = uploaded.getvalue().decode("utf-8", errors="ignore")

    st.subheader("청킹 설정")
    c1, c2 = st.columns(2)
    chunk_size = c1.slider("청크 크기 (문자 수)", 200, 1000, 500, step=50)
    overlap = c2.slider("겹침 (overlap)", 0, 200, 50, step=10)
    chunks = chunk_text(raw_text, chunk_size, overlap)
    if not chunks:
        st.warning("문서에서 추출한 텍스트가 비어 있습니다.")
        return
    st.caption(f"청크 {len(chunks)}개 생성됨")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("context"):
                with st.expander("📄 출처 청크 보기"):
                    st.markdown(m["context"])

    if prompt := st.chat_input("이 문서에 대해 궁금한 것을 물어보세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # [핵심] st.status로 검색→답변 단계를 보여준다 — 판단(질문=쿼리)→행동(검색)→관찰(LLM 전달)
            with st.status("문서에서 답을 찾는 중...", expanded=True) as status:
                st.write("1) 질문을 검색어로 관련 청크 찾는 중...")
                matches = retrieve(prompt, chunks, embed_provider)
                st.write(f"2) 상위 {len(matches)}개 청크 확보 → LLM에게 전달 중...")
                context = assemble_context(matches)
                answer = answer_with_context(prompt, context, llm_provider)
                status.update(label="완료", state="complete", expanded=False)
            st.markdown(answer)
            with st.expander("📄 출처 청크 보기"):
                st.markdown(context)
        st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})


if __name__ == "__main__":
    main()
