# 미니프로젝트 연동 — "나만의 지식DB" 라이트 RAG (무키·무네트워크)
# [왜] 19강 미니 RAG는 chromadb·sentence-transformers 임베딩(의미 기반 유사도)으로 검색했다.
#      이 앱은 TF-IDF(단어 빈도 기반 유사도)로 같은 "질문→벡터화→유사도 검색" 흐름을 재현한다 —
#      무거운 임베딩 모델·API 키 없이 배포 URL에서도 바로 동작하는 가벼운 버전이다.
# [흐름] search_docs()는 UI·st 의존이 없는 순수 함수다 — 다만 apps/rag_chatbot.py의 retrieve()는
#      (인덱스, 청크, 유사도) 튜플이 필요해 이 함수 대신 _top_matches()를 직접 재사용한다(본문 미사용).
#      TOOL_FUNCS에 그대로 등록하면 "내 문서를 검색하는 도구"가 된다(도전 과제·apps/m5b_agent_loop.py 패턴 참고).
# 실행: python3.11 -m streamlit run apps/rag_lite.py

import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [왜] matplotlib 기본 폰트(DejaVu Sans)엔 한글 글리프가 없어 라벨이 □□로 깨진다(m3_penguins 패턴 복사).
from matplotlib import font_manager

for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
    if any(_font.name == _f for _font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1F4E79"
TEAL = "#0E6B56"
CORAL = "#993C1D"


# ── ① 청킹 + 검색 — UI·st 의존 없는 순수 함수 (ST4 도구로 import 가능) ──
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """[핵심] 문자 단위 슬라이딩 윈도우 청킹. overlap만큼 겹쳐 잘라 청크 경계의 문맥 단절을 줄인다."""
    text = text.strip()
    if not text:
        return []
    # [안전] overlap이 chunk_size에 근접하면 step이 0에 가까워져 청크가 폭발적으로 늘어난다 —
    #        최소 step 20자로 방어(슬라이더 조합 200/200 같은 극단값에서도 앱이 멈추지 않게).
    step = max(chunk_size - overlap, 20)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


@st.cache_resource
def build_index(chunks: tuple[str, ...]):
    """[왜] cache_resource — TfidfVectorizer는 학습된 리소스라, 같은 청크 조합이면 재학습 없이 재사용한다.
    🔬 심화: analyzer='char_wb', ngram_range=(2,3)으로 바꾸면 형태소 분석기 없이도 한국어 부분일치가 더 강해진다."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def _top_matches(query: str, chunks, vectorizer, matrix, top_k: int = 3):
    """질문을 같은 벡터공간으로 변환해 코사인 유사도로 상위 top_k (인덱스, 청크, 유사도)를 반환."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [(int(i), chunks[i], float(sims[i])) for i in top_idx]


def search_docs(query: str, chunks, vectorizer, matrix) -> str:
    """[도전 과제 전용 — 본문(rag_chatbot.py)에서는 호출하지 않음] 순수 함수 — top-3 청크를 이어붙인
    문자열을 반환한다. 본문 retrieve()는 (인덱스, 청크, 유사도) 튜플이 필요해 이 함수 대신 _top_matches()를
    직접 쓴다. st 의존이 없어 TOOL_FUNCS에 그대로 등록하면 "내 문서를 검색하는 도구"가 된다
    (apps/m5b_agent_loop.py 패턴) — 도전 과제에서 개별 도구로 등록해볼 때 이 함수를 쓴다."""
    if not chunks:
        return "검색할 문서가 없습니다."
    matches = _top_matches(query, chunks, vectorizer, matrix, top_k=3)
    return "\n\n".join(f"[청크 {i}] (유사도 {score:.2f})\n{chunk}" for i, chunk, score in matches)


# ── ② 임베딩 검색 — API 임베딩으로 확장 (ST4 apps/rag_chatbot.py가 재사용) ──
# [왜] 위 TF-IDF는 "단어가 얼마나 겹치는가"만 본다. 임베딩(embedding)은 다른 단어라도 의미가
#      비슷하면 가깝게 인식한다(19강 sentence-transformers와 같은 개념). 아래 EMBED_PROVIDERS는
#      apps/m5b_agent_loop.py의 PROVIDERS(20강 LLM provider 규약)와 똑같은 모양이다 — local(Ollama)·
#      openrouter·openai 셋 다 OpenAI 호환 /v1/embeddings 엔드포인트를 지원해서 client.embeddings.create(...)
#      하나로 base_url만 바꿔가며 세 곳을 전환할 수 있다(실측: Ollama 0.32, nomic-embed-text 768차원).
EMBED_PROVIDERS = {
    "local": {"base_url": "http://localhost:11434/v1", "model": "nomic-embed-text", "key": "ollama"},  # 무료 — 사전과제: ollama pull nomic-embed-text
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3-embedding-8b", "key_env": "OPENROUTER_API_KEY"},  # $0.01/1M 토큰 — 실측 최저가+한국어 우수
    "openai": {"base_url": None, "model": "text-embedding-3-small", "key_env": "OPENAI_API_KEY"},  # $0.02/1M 토큰
}


def _embed_key(provider: str) -> str:
    cfg = EMBED_PROVIDERS[provider]
    if "key" in cfg:
        return cfg["key"]
    return os.environ.get(cfg["key_env"], "")


def embed_provider_available(provider: str) -> bool:
    """[대칭] apps/m5b_agent_loop.py의 provider_available과 같은 판단 로직 —
    local은 Ollama 서버 응답 여부, 나머지는 키 존재 여부로 판단한다."""
    if provider == "local":
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
            return True
        except Exception:
            return False
    return bool(_embed_key(provider))


@st.cache_resource
def _embed_client(provider: str):
    from openai import OpenAI
    cfg = EMBED_PROVIDERS[provider]
    return OpenAI(base_url=cfg["base_url"], api_key=_embed_key(provider) or "none")


@st.cache_data(show_spinner="임베딩 계산 중...")
def embed_texts(chunks: tuple[str, ...], provider: str) -> list[list[float]]:
    """[캐시] 청크 튜플 + provider가 같으면 재호출 없이 캐시된 벡터를 재사용한다
    (cache_data 키 = 인자 전체 — chunk_size·overlap을 바꿔 청크가 달라지면 자동으로 다시 계산됨).
    반환: 청크 개수만큼의 임베딩 벡터 리스트."""
    client = _embed_client(provider)
    model = EMBED_PROVIDERS[provider]["model"]
    resp = client.embeddings.create(model=model, input=list(chunks))
    return [d.embedding for d in resp.data]


def search_docs_embed(query: str, chunks, chunk_vectors, provider: str, top_k: int = 3):
    """임베딩 기반 top_k 검색 — 위 _top_matches(TF-IDF)와 같은 반환 형식 (인덱스, 청크, 유사도)."""
    client = _embed_client(provider)
    model = EMBED_PROVIDERS[provider]["model"]
    q_vec = client.embeddings.create(model=model, input=[query]).data[0].embedding
    sims = cosine_similarity([q_vec], chunk_vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [(int(i), chunks[i], float(sims[i])) for i in top_idx]


def _extract_pdf_text(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main():
    st.set_page_config(page_title="나만의 지식DB — 라이트 RAG", page_icon="📚", layout="centered")
    st.title("📚 나만의 지식DB — 라이트 RAG")
    st.caption("txt·md·pdf 문서를 올려 청킹 → TF-IDF 검색까지 무키·무네트워크로 체감하는 미니 RAG")

    with st.expander("💡 이전 챕터와 연결"):
        st.markdown(
            "- **19강 미니 RAG**: chromadb·sentence-transformers **임베딩**(의미 기반 유사도)으로 "
            "검색했습니다. 이 앱은 **TF-IDF**(단어 빈도 기반 유사도)로 같은 '질문→벡터화→유사도 검색' "
            "흐름을 재현합니다 — 무거운 임베딩 모델 없이 배포 URL에서도 바로 동작하는 가벼운 버전입니다.\n"
            "- **ST4 에이전트 도구**: `search_docs()`를 `TOOL_FUNCS`에 등록하면 '내 문서 검색' 도구가 됩니다."
        )

    uploaded = st.file_uploader("문서 업로드 (txt·md·pdf)", type=["txt", "md", "pdf"])
    if uploaded is None:
        st.info("문서를 업로드하면 청킹 슬라이더와 검색을 체험할 수 있습니다.")
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
    with st.expander("청크 샘플 보기"):
        for i, c in enumerate(chunks[:3]):
            st.text(f"[{i}] {c[:200]}{'...' if len(c) > 200 else ''}")

    vectorizer, matrix = build_index(tuple(chunks))

    st.subheader("질문 검색")
    query = st.text_input("이 문서에 대해 궁금한 것을 물어보세요")
    if query:
        matches = _top_matches(query, chunks, vectorizer, matrix, top_k=3)
        for rank, (i, chunk, score) in enumerate(matches, start=1):
            st.markdown(f"**{rank}위 — 청크 {i}** (유사도 {score:.2f})")
            st.write(chunk)

        colors = [NAVY, TEAL, CORAL]
        fig, ax = plt.subplots()
        labels = [f"청크 {i}" for i, _, _ in matches][::-1]
        scores = [score for _, _, score in matches][::-1]
        ax.barh(labels, scores, color=colors[:len(matches)][::-1])
        ax.set_xlabel("코사인 유사도")
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
