# [기술 보고서] LLM Multi-Agent 기반 금융 RAG 및 포트폴리오 최적화 시스템

## 1. 시스템 개요 (Executive Summary)
본 시스템은 뉴스 데이터와 매크로 시나리오를 결합하여 최적의 자산 배분 가이드를 제공하는 **Self-Improving Financial RAG 시스템**입니다.  
단순 검색(Retrieval)을 넘어, **상승론(Bull)과 하락론(Bear)의 논리적 경합(Debate)**을 통해 시장의 괴리(Divergence)를 분석하고, 그 결과를 바탕으로 퀀트 모델의 파라미터를 미세 조정(Fine-tuning)한 뒤 수학적 최적화(SLSQP)를 수행하는 **End-to-End 파이프라인**을 구축하였습니다.

---

## 2. 핵심 아키텍처 및 알고리즘 설명

### 2.1 Multi-Agent 추론 워크플로우 (LangGraph 활용)
단일 LLM의 편향성을 제거하기 위해 LangGraph를 이용한 상태 중심(Stateful) 에이전트 구조를 설계하였습니다.

- **Debate Node**: 뉴스 컨텍스트를 바탕으로 ‘상승론자’와 ‘하락론자’ 페르소나가 각각 독립적인 분석 수행
- **Judge Node (CIO)**: 두 의견을 종합하여 최종 합의문(Consensus)을 도출하고 시장 트렌드 및 리스크 점수 산출
- **Evaluator Node (Feedback Loop)**: 생성된 결론이 질문의 요지 및 원본 데이터와 일치하는지 검증  
  - 점수 미달 시(8점 미만) 토론 단계로 회귀하여 논리 보완(Self-Refinement)

---

### 2.2 하이브리드 RAG 전략
- **Vector DB**: FAISS를 활용하여 *Deleveraging*, *Goldilocks* 등 과거 핵심 경제 시나리오를 인덱싱
- **Dynamic Search**: 단순 사용자 질문이 아닌, 에이전트가 도출한 **최종 합의문**을 검색 쿼리로 사용  
  - 현재 상황과 가장 유사한 과거 매크로 앵커(Anchor) 데이터 추출

---

### 2.3 Quant Engine & Optimization
- **Parameter Tuning**: RAG로 추출된 과거 통계값(Expected Return, Volatility)을 LLM이 현재 시장 상황에 맞춰 보정
- **SLSQP Optimizer**: `scipy.optimize`를 사용하여 다음 제약 조건을 만족하는 샤프 지수 기반 최적 포트폴리오 산출
  - 각 자산 비중 제한: 0~45%
  - 총합 제약: 1.0

---

## 3. 주요 요구사항 구현 상세

| 요구사항 | 구현 내용 |
|---|---|
| 1. 데이터 전처리 | `SCENARIO_KB`를 통해 구조화된 금융 시나리오 데이터 구축 및 임베딩 처리 |
| 2. 벡터 DB 색인 | `langchain_community.vectorstores.FAISS`를 활용한 고속 벡터 검색 구현 |
| 3. 관련 문서 검색 | `similarity_search`를 통해 현재 시장 상황과 매칭되는 시나리오 메타데이터 추출 |
| 4. LLM 답변 생성 | local-llama 모델 활용, Structured Output(Pydantic) 기반 정형 데이터 생성 |
| 5. 정확성 평가/개선 | Evaluator 노드를 통한 반복 추론(Iterative Refinement) 로직 및 조건부 엣지 구현 |
| 6. REST API 구축 | FastAPI 기반 `/analyze`(분석), `/metrics`(성능 측정) 엔드포인트 제공 |
| 7. 성능 측정 | `PerformanceTracker` 클래스로 Latency, Eval Score, Retry Rate 실시간 집계 |

---

## 4. 시스템 동작 알고리즘 (Flow Chart)
1. **Input**: 사용자의 금융 질문  
   - 예: “현재 금리 인하 기대감이 시장에 미치는 영향은?”
2. **Debate**: 뉴스 기반 Bull/Bear 의견 대립 생성
3. **Judgement**: CIO 에이전트가 시장 심리(Sentiment) 및 가격 괴리 분석
4. **Evaluation**: 비판적 검토  
   - Pass 시 다음 단계 진행 / Fail 시 Debate 재수행
5. **Retrieval**: 합의된 내용을 키워드로 FAISS에서 유사 시나리오 로드
6. **Estimation**: LLM이 시나리오 통계치와 현재 지표를 결합해 기대수익률 및 공분산 추정  
   - $$\mu$$ (기대수익률), $$\Sigma$$ (공분산)
7. **Optimization**: SLSQP 알고리즘으로 최적 자산 비중 산출  
   - $$w^*$$
8. **Output**: 최종 투자 뷰(Manager View) 및 포트폴리오 가이드 반환

---

## 5. 기술 스택 (Tech Stack)
- **Language**: Python 3.10+
- **Frameworks**: LangChain, LangGraph, FastAPI
- **LLM**: Local Llama (OpenAI-compatible API via)
- **Database**: FAISS (Vector DB)
- **Math/Stats**: NumPy, SciPy (SLSQP), Pydantic
- **DevOps**: Uvicorn, Python-dotenv

---

## 6. 성능 측정 및 개선 결과
시스템 내부의 `PerformanceTracker`를 통해 다음 지표를 관리합니다.

- **평균 응답 시간(Latency)**: Multi-agent 토론 과정으로 단일 호출보다 시간은 소요되나 논리적 완성도 확보
- **재시도율(Retry Rate)**: 평가 노드로 초기 논리 모순을 발견/수정하여 최종 답변 신뢰도(Reliability) 향상
- **정확도(Eval Score)**: 금융 전문가 페르소나 기반 자가 채점으로 지속적인 프롬프트 개선 가능

---

## [부록] 실행 방법

1. `.env` 파일에 필요한 환경 변수 설정
2. 로컬 LLM 서버(Port: 8090) 실행 확인
3. `python main.py` 실행 후 `http://localhost:8088/docs` 접속하여 API 테스트

(참고) 데이터 소스
금융 뉴스: LS 증권 Open API 기반 WebSocket 수신: https://openapi.ls-sec.co.kr/howto-sample

### 로컬 LLM 서버 실행 예시
````bash
.\llama-server -m "V:\PythonProject\hf_cache_gguf\Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --port 8090 --host 0.0.0.0 -ngl 99 -c 8196 -fa auto --embedding --pooling mean


Request body:
{
  "question": "현재 캐빈 워시가 연준 의장으로 지명된 상황에 한국 옵션 시장에 미치는 영향은?",

}

