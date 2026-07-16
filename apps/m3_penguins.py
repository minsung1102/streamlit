# ST2 — 시각화·EDA 대시보드: 팔머펭귄 종 분류기
# [왜] 노트북은 만든 사람만 본다. 슬라이더로 트리 개수·깊이를 바꿔가며 정확도·혼동행렬이
#      실시간으로 바뀌는 것을 "체감"하는 순간, 시각화는 커뮤니케이션 도구가 된다.
# [흐름] 12~13강에서 배운 혼동행렬(confusion matrix)·과적합(overfitting) 개념의 웹 버전
# [왜] 사이드바 라디오로 RandomForest(여러 트리 평균) vs DecisionTree(트리 1개)를 실시간
#      비교한다 — 같은 슬라이더 값에서 두 모델의 train/test 격차가 어떻게 달라지는지 직접 본다.
# 실행: python3.11 -m streamlit run apps/m3_penguins.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# [왜] matplotlib 기본 폰트(DejaVu Sans)엔 한글 글리프가 없어 "예측/실제" 같은 라벨이 □□로 깨진다.
#      로컬은 Mac(AppleGothic)이 있어 멀쩡하지만, Streamlit Cloud(Linux)엔 한글 폰트가 없어 배포 때 깨진다.
#      → packages.txt의 fonts-nanum으로 Cloud에 NanumGothic을 설치하고, 아래에서 설치된 폰트를 순서대로 찾는다.
from matplotlib import font_manager

for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
    if any(_font.name == _f for _font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 마이너스 기호(−)가 깨지는 것 방지

# [왜] 4개 수치형 특징만 모델 입력으로 쓴다 — species·island·sex 등 범주형은
#      이번 데모의 스코프 밖(간단한 4특징 분류기로 유지).
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


# [왜] cache_data — 데이터는 호출마다 "복사본"을 돌려준다. 화면 곳곳에서 df를 자유롭게
#      다뤄도(정렬·필터) 원본 캐시가 오염되지 않는다. (위 4절에서 배운 그 데코레이터)
@st.cache_data
def load_data():
    return sns.load_dataset("penguins").dropna()


# [왜] cache_resource — 모델은 "리소스"라 싱글턴으로 공유한다. 슬라이더가 같은 값 조합으로
#      돌아오면(예: 옆 탭 갔다가 다시 옴) 재학습 없이 캐시된 모델을 그대로 재사용한다.
# [흐름] 344행짜리 데이터라 어차피 수십 ms지만, 100만 행이면 이 캐시가 응답속도를 가른다.
# [통계] train_model은 st 부작용 없는 순수 함수 — df를 인자로 받아 테스트에서 바로 호출 가능.
@st.cache_resource
def train_model(df, model_type, n_estimators, max_depth):
    df = df.dropna()
    # [왜] sklearn 분류기는 문자열 라벨을 직접 못 받는다 — species 텍스트를 숫자
    #      인덱스로 바꿔야(LabelEncoder) fit/predict에 넣을 수 있다.
    le = LabelEncoder()
    X = df[FEATURES]
    y = le.fit_transform(df["species"])
    # [왜] stratify=y — 종(species)별 비율을 train/test에 동일하게 유지한다.
    #      층화 없이 나누면 소수 종이 test에 몰리거나 아예 빠질 수 있다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # [왜] model_type으로 RandomForest(여러 트리 평균)와 DecisionTree(트리 1개)를 전환한다 —
    #      DecisionTree는 n_estimators 개념이 없어 max_depth만 반영한다.
    if model_type == "DecisionTree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    model.fit(X_train, y_train)

    # [왜] train_acc까지 계산해야 "과적합 체감"이 가능하다 — test_acc만 보면
    #      train과의 격차(=과적합 신호)를 눈으로 볼 수 없다.
    train_acc = accuracy_score(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # [왜] 단일 split의 test_acc는 seed에 따라 흔들리는 점추정일 뿐이다 — 5-fold 층화
    #      교차검증의 평균±표준편차가 "이 정확도를 얼마나 믿을 수 있는지"를 함께 보여준다.
    #      (🔬 심화 — 화면에는 접은 패널 안에서만 노출한다)
    if model_type == "DecisionTree":
        cv_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        cv_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    cv_scores = cross_val_score(
        cv_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # [왜] impurity 기반 feature_importances_는 상관된 특징(flipper·body_mass, 상관 0.87)
    #      끼리 중요도를 임의로 나눠 가져 왜곡된다 — held-out permutation importance가
    #      "이 특징을 섞으면 정확도가 얼마나 떨어지는가"를 직접 재서 더 정직하다.
    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)
    perm_importance = dict(zip(FEATURES, perm.importances_mean))

    return {
        "model": model,
        "le": le,
        "cm": cm,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "perm_importance": perm_importance,
    }


def main():
    # [왜] set_page_config는 스크립트 최상단에서 한 번만 호출 가능 — layout="wide"로
    #      탭·컬럼이 좁은 화면에서도 잘리지 않게 미리 넓게 잡아둔다.
    st.set_page_config(page_title="펭귄 종 분류기", page_icon="🐧", layout="wide")
    st.title("🐧 팔머펭귄 종 분류 대시보드")
    st.caption("RandomForest + Streamlit — 12~13강 혼동행렬·과적합 개념의 웹 버전")

    df = load_data()

    # [왜] 모델 설정을 사이드바에 몰아두면 본문(EDA·성능·예측 탭)이 파라미터 조작과
    #      분리돼 화면이 덜 어수선해진다 — 데이터 vs 조작을 시선으로 구분.
    st.sidebar.header("⚙️ 모델 설정")
    # ✏️ [비교 토글] RandomForest(여러 트리 평균) vs DecisionTree(트리 1개) — 같은 슬라이더
    #    값에서 두 모델의 train/test 격차가 어떻게 달라지는지 실시간으로 비교해보세요.
    # [왜] horizontal=True로 라디오를 가로로 배치 — 사이드바 폭이 좁아 세로 배치면
    #      두 옵션 이름이 줄바꿈되어 읽기 불편하다.
    model_type = st.sidebar.radio("모델 종류", ["RandomForest", "DecisionTree"], horizontal=True)
    # [왜] disabled=로 DecisionTree 선택 시 슬라이더를 잠근다 — 존재하지 않는 개념(트리 개수)을
    #      조작 가능한 것처럼 보이게 두면 학생이 "왜 값을 바꿔도 안 변하지?" 헷갈린다.
    n_estimators = st.sidebar.slider(
        "트리 개수 (n_estimators)", 10, 200, 100, step=10,
        disabled=(model_type == "DecisionTree"),
    )
    if model_type == "DecisionTree":
        st.sidebar.caption("단일 트리는 '트리 개수' 개념이 없어 max_depth만 반영됩니다.")
    # [왜] max_depth는 두 모델 공통 파라미터라 radio 분기 밖에 둔다 — 값이 5 근처를
    #      넘으면 n_estimators 변화가 화면에 거의 드러나지 않는다(위 실습 안내 참고).
    max_depth = st.sidebar.slider("최대 깊이 (max_depth)", 2, 10, 5)
    # ✏️ [학생 실습 지점] 슬라이더 값을 극단으로 바꿔보세요 — 트리 10개·깊이 2 vs 트리 200개·깊이 10.
    #    Train 정확도는 계속 올라가는데 Test 정확도는 안 따라오는 격차가 "과적합 감각"입니다.
    #    ⚠️ max_depth≥5에서는 n_estimators 단독 조정이 화면에 거의 영향을 주지 않습니다 —
    #    depth를 먼저 바꿔보세요.

    # [왜] train_model()이 cache_resource라 슬라이더가 이전과 같은 값 조합으로 돌아오면
    #      재학습 없이 캐시된 결과를 즉시 반환한다 — 여기서 초 단위 지연을 아낀다.
    out = train_model(df, model_type, n_estimators, max_depth)
    model, le, cm = out["model"], out["le"], out["cm"]

    # [왜] EDA·모델 성능·예측을 탭으로 나눈다 — 세 관심사를 한 화면에 몰아넣으면
    #      스크롤이 길어지고 "지금 뭘 보는 화면인지" 놓치기 쉽다.
    tab_eda, tab_perf, tab_pred = st.tabs(["📊 EDA", "🌲 모델 성능", "🔮 예측"])

    with tab_eda:
        st.subheader("데이터 미리보기")
        # [왜] 모델을 보여주기 전에 원본 표부터 보여준다 — "숫자가 어디서 왔는지" 먼저
        #      확인해야 이후 시각화·정확도 숫자를 믿고 해석할 수 있다.
        st.dataframe(df)

        st.subheader("두 변수 관계 살펴보기")
        # [왜] 두 selectbox를 columns(2)로 나란히 배치 — X·Y축을 자유롭게 조합해보며
        #      어떤 특징 쌍이 종을 가장 잘 갈라놓는지 학생이 직접 탐색하게 한다.
        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("X축", FEATURES, index=0)
        y_axis = col2.selectbox("Y축", FEATURES, index=2)
        # [왜] hue="species"로 색을 종에 맞춰 자동 분리 — 점들이 뭉쳐 있으면 그 두 특징만으로는
        #      종을 가르기 어렵다는 뜻이고, 갈라져 있으면 좋은 특징 조합이라는 뜻이다.
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="species", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with tab_perf:
        st.subheader("모델 정확도 — Train vs Test")
        # [왜] metric의 delta=로 Test-Train 격차를 화살표+색으로 바로 보여준다 —
        #      격차가 크게 음수(빨강 ↓)로 나오면 과적합(overfitting) 신호다.
        c1, c2 = st.columns(2)
        c1.metric("Train 정확도", f"{out['train_acc']:.1%}")
        c2.metric(
            "Test 정확도",
            f"{out['test_acc']:.1%}",
            delta=f"{(out['test_acc'] - out['train_acc']):+.1%}",
        )

        st.subheader("혼동행렬 (Confusion Matrix)")
        # [흐름] 13강에서 "오분류 이미지를 눈으로 확인하라"고 배운 것의 표 버전 —
        #        어느 종끼리 헷갈리는지가 정확도 숫자 하나보다 훨씬 많은 정보를 준다.
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
        )
        # [왜] x축=예측, y축=실제로 라벨링 — sklearn confusion_matrix 기본 축 순서와
        #      맞춰야 "대각선=맞춘 것"이라는 해석이 어긋나지 않는다.
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        st.pyplot(fig)
        plt.close(fig)

        # [왜] 교차검증·특징중요도는 심화 내용이라 expander(접이식) 안에 숨겨둔다 —
        #      "못 열어봐도 정상", 기본 화면은 핵심 지표(정확도·혼동행렬)만 보이게 한다.
        with st.expander("🔬 심화: 교차검증 신뢰도 + 특징 중요도 더 알아보기"):
            st.markdown("**교차검증(cross-validation) 정확도 — 이 숫자를 얼마나 믿을 수 있나**")
            st.metric("5-fold 교차검증 평균", f"{out['cv_mean']:.1%} ± {out['cv_std']:.1%}")
            st.caption(
                "단일 split의 test_acc는 seed에 따라 흔들리는 점추정 — 5-fold 층화(StratifiedKFold)로 "
                "5번 다시 나눠 평균±표준편차를 보면 이 정확도를 얼마나 믿을 수 있는지 알 수 있다."
            )

            st.markdown("**특징 중요도 (Permutation Importance)**")
            perm_df = pd.DataFrame(
                {"feature": FEATURES, "importance": [out["perm_importance"][f] for f in FEATURES]}
            ).set_index("feature")
            st.bar_chart(perm_df)
            st.caption(
                "중요도=예측 기여, 인과 아님. impurity 대신 held-out permutation을 쓴다. "
                "단 permutation도 상관 특징끼리(flipper·body_mass)는 한쪽을 섞어도 나머지로 "
                "모델이 맞혀 둘 다 과소평가될 수 있다 — 완벽한 도구가 아니라 impurity보다 정직한 근사다."
            )
            st.caption(
                "⚠️ 상관 0.87은 세 종을 합쳤을 때 값 — 종 내부에서는 0.47~0.71로 더 약하다 "
                "(Simpson's paradox: 그룹을 합치면 관계가 과장돼 보일 수 있음)."
            )

        # [왜] 두 모델을 비교(toggle)만 하고 끝내지 않는다 — "그래서 실무에선 어느 쪽?"에
        #      답하는 한 줄을 항상 화면 하단에 남긴다(교육 데모 원칙④).
        st.caption(f"현재 모델: **{model_type}**")
        st.caption(
            "→ 실무에서는 기본값으로 RandomForest를 쓰고, '왜 이렇게 예측했는지'를 "
            "트리 1개로 설명해야 할 때만 DecisionTree로 바꿉니다."
        )

    with tab_pred:
        st.subheader("새 펭귄 데이터로 예측")
        # [왜] 4개 입력을 columns(2)로 2개씩 묶는다 — c1은 부리(길이·깊이), c2는
        #      몸통(물갈퀴·체중) 관련 값이라 화면 세로 길이도 줄이고 관련 값끼리 붙여 보여준다.
        c1, c2 = st.columns(2)
        bill_length = c1.number_input("부리 길이 (mm)", 30.0, 60.0, 45.0)
        bill_depth = c1.number_input("부리 깊이 (mm)", 13.0, 22.0, 17.0)
        flipper_length = c2.number_input("물갈퀴 길이 (mm)", 170.0, 235.0, 200.0)
        body_mass = c2.number_input("체중 (g)", 2700.0, 6300.0, 4200.0)

        # [흐름] ST1에서 배운 rerun 모델 그대로 — number_input 하나만 바꿔도 스크립트 전체가
        #        다시 실행되며 predict_proba가 즉시 갱신된다. 별도 "제출" 버튼이 필요 없다.
        X_new = pd.DataFrame(
            [[bill_length, bill_depth, flipper_length, body_mass]], columns=FEATURES
        )
        proba = model.predict_proba(X_new)[0]
        pred_idx = proba.argmax()
        pred_species = le.classes_[pred_idx]
        confidence = proba[pred_idx]

        st.success(f"예측 종: **{pred_species}** (확신도 {confidence:.1%})")

        # [도전 과제 반영] 확신도가 낮으면 "애매한 경계 사례"임을 알려준다. 단 이 값은 보정된
        #      확률이 아니라 트리 투표 비율이라(아래 캡션), 70%라는 경계선은 절대 기준이 아니라 참고용 임계다.
        if confidence < 0.7:
            st.warning(
                "⚠️ 확신도가 70% 미만입니다 — 입력값이 애매한 경계 사례일 수 있습니다"
                "(70%는 통계적 보증이 아니라 참고용 임계선)."
            )

        # [왜] 1등 종의 확률만 보여주지 않고 3종 전체를 막대그래프로 — 2등과의 격차가
        #      크지 않으면 "확신도 97%"도 사실 애매한 경계일 수 있음을 함께 드러낸다.
        proba_df = pd.DataFrame({"species": le.classes_, "probability": proba}).set_index(
            "species"
        )
        st.bar_chart(proba_df)
        st.caption("트리 투표 비율일 뿐 보정된 확률 아님 — '97%면 97% 맞다'는 뜻 아님")


# [왜] if __name__ == "__main__": 가드 — 이 파일을 나중에 다른 스크립트에서
#      import해도 main()이 자동 실행되지 않도록 막아둔다.
if __name__ == "__main__":
    main()
