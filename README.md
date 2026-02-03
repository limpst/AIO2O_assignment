# 📈 AIO2O assignment: Intelligence-Driven Financial RAG & Portfolio Optimizer

실시간 금융 뉴스 데이터와 거시 경제 지표를 결합해 최적의 투자 전략을 도출하는 금융 특화 **RAG(Retrieval-Augmented Generation)** 시스템입니다.  
단순 요약을 넘어, **Multi-Agent 기반 Debate 메커니즘**으로 시장 편향을 줄이고 **퀀트 모델(SLSQP)**로 최종 자산 배분안을 제시합니다.

## 🏗️ 시스템 아키텍처 및 주요 기능

### 1) 지능형 데이터 파이프라인 (Data & Indexing)
- **실시간 수집**:  
  `yfinance`로 KOSPI200, S&P500 등 매크로 지표를 수집하고, MySQL DB에 웹소켓으로 적재되는 라이브 뉴스를 통합합니다.
- **Context 가공**:  
  `RecursiveCharacterTextSplitter` + `ParentDocumentRetriever`로 단순 문장 단위가 아닌 **금융 맥락(Context)**이 보존된 청킹 전략을 적용합니다.
- **데이터 전처리**:  
  단순 가격이 아닌 **로그 수익률** $$\ln(P_t / P_{t-1})$$ 및 **시차(Lag) 조정**으로 모델 분석 효율을 극대화합니다.

### 2) Multi-Agent Debate 시스템
시장 상황을 다각도에서 분석하기 위해 3개의 독립 에이전트가 협업합니다.
- **🐂 Bull Agent**: 상승 동력 및 긍정 지표(낙관론) 중심 분석
- **🐻 Bear Agent**: 리스크 및 하락 시나리오(신중론) 중심 분석
- **⚖️ Consensus Judge (CIO)**:  
  두 의견을 검토하고 **Price Action vs 뉴스 심리** 괴리를 점검해 최종 투자 판단을 확정

### 3) Divergence(괴리) 관리 및 가드레일
- **Price–Sentiment Check**:  
  뉴스는 호재인데 가격이 하락하는 경우 등, 심리와 가격 흐름의 괴리를 `divergence_note`로 관리합니다.
- **Decision Priority**:  
  시스템은 뉴스 심리보다 **실제 가격 흐름(Price Action)**을 우선하도록 설계되어 감정적 과열을 억제합니다.

### 4) 퀀트 포트폴리오 최적화 (SLSQP)
최종 시장 뷰(`bullish`, `bearish`, `neutral`)에 따라 자산 배분을 실행합니다.
- **목표 함수**: 기대수익률 벡터 $$\mu$$ 와 변동성·상관계수를 고려한 투자 효용 극대화
- **제약 조건**: 총 자본금(예: 2억 원) 내에서 옵션(Call/Put) 및 선물(Future) 비중 최적화
- **알고리즘**: `scipy.optimize.minimize` (SLSQP) 기반 비선형 제약 최적화

---

## 📊 데이터 흐름 (Data Flow)
1. **Query Expansion**: 사용자 질문을 “Bear steepening” 등 금융 전문 용어를 포함한 최적화 쿼리로 재작성  
2. **Hybrid Retrieval**: 벡터 검색(의미론) + SQL 검색(날짜/키워드) 결합으로 최적 뉴스 추출  
3. **Agent Debate**: 상반된 시각 토론을 거쳐 객관적인 매니저 뷰 생성  
4. **Portfolio Execution**: 확정된 뷰 기반으로 실제 투자 수량(Qty) 산출  
5. **Self-Learning**: 수익률 기반 성공 시나리오 파라미터를 DB에 피드백하여 고도화

---

## 🚀 실행 방법

### 서버 실행
````bash
# FastAPI 서버 가동
python main.py

## 📚 API Docs
- http://127.0.0.1:8088/docs` 접속 시 **Swagger** 인터페이스 확인 가능



## 📰 뉴스 데이터 예시 (Internal Structure)
````python
# 실제 라이브 코드에서는 웹소켓을 통해 실시간 DB 적재
test_news = [
    {"date": "2026-02-03", "time": "20:41", "title": "워시 연준 의장 후보, 적극적 금리 인하 추진 관측"},
    {"date": "2026-02-03", "time": "16:35", "title": "코스피 사상 최고치 경신, 외국인 대거 순매수"}
]

## 📈 투자 전략 예시 (Bullish Case)

시장이 상승세일 때, 시스템은 다음과 같은 기대수익률 벡터 $$\mu$$를 기반으로 최적 비중을 도출합니다.

$$\mu_{bullish} = [0.08, -0.02, -0.06, 0.03, 0.12]$$

- **자산 순서**: `(Call_L, Call_S, Put_L, Put_S, Future)`
