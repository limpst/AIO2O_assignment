import os
import json
import warnings
import uvicorn
import numpy as np
from typing import TypedDict, List, Dict, Optional, Any, Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# Optimization
from scipy.optimize import minimize

import time # 성능 측정을 위해 추가
from collections import deque # 통계 저장을 위해 추가

# ---------------------------------------------------------
# 1. 환경 설정 및 LLM 초기화
# ---------------------------------------------------------
load_dotenv(override=True)
warnings.filterwarnings("ignore")

# 로컬 Llama 서버 설정
llm = ChatOpenAI(
    model="local-llama",
    base_url="http://localhost:8090/v1",
    api_key="no-key-needed",
    temperature=0.0,
    timeout=600
)

embeddings = OpenAIEmbeddings(
    model="local-model",
    base_url="http://localhost:8090/v1",
    api_key="no-key-needed"
)

# ---------------------------------------------------------
# 2. RAG 지식 베이스 (Scenario KB -> FAISS)
# ---------------------------------------------------------
SCENARIO_KB = [
    {
        "id": "EXTREME_BEAR_0000",
        "name": "Deleveraging",
        "desc": "마진콜 및 부채 축소로 인한 강제 매도 장세. 하락 변동성 극대화 및 상관관계 수렴.",
        "mu": [-0.3385, 0.1038, 0.5128, -0.1338, -0.2469],
        "vol": [0.3599, 0.34, 0.4803, 0.4954, 0.4017],
        "corr": [
            [1.0, 0.7718, -0.1502, 0.0634, 0.6265],
            [0.7718, 1.0, 0.0574, -0.0311, 0.0945],
            [-0.1502, 0.0574, 1.0, 0.7518, -0.53],
            [0.0634, -0.0311, 0.7518, 1.0, -0.10],
            [0.6265, 0.0945, -0.53, -0.10, 1.0]
        ],
    },
    {
        "id": "BULLISH_0001",
        "name": "Goldilocks",
        "desc": "저물가·적정 성장 속 이상적인 우상향. 낮은 변동성 및 콜 옵션 수익성 개선.",
        "mu": [0.1676, -0.0683, -0.2374, 0.0894, 0.1393],
        "vol": [0.1356, 0.1328, 0.1848, 0.1731, 0.1207],
        "corr": [
            [1.0, 0.7992, -0.0815, 0.0967, 0.6250],
            [0.7992, 1.0, 0.0907, -0.0271, 0.0767],
            [-0.0815, 0.0907, 1.0, 0.8390, -0.52],
            [0.0967, -0.0271, 0.8390, 1.0, -0.10],
            [0.6250, 0.0767, -0.52, -0.10, 1.0]
        ],
    },
]


def _build_scenario_vectorstore():
    docs = []
    for s in SCENARIO_KB:
        text = f"시나리오명: {s['name']}\n상황설명: {s['desc']}"
        docs.append(Document(page_content=text, metadata=s))
    return FAISS.from_documents(docs, embeddings)


scenario_vs = _build_scenario_vectorstore()


# ---------------------------------------------------------
# 0. 성능 통계 저장소 (In-memory)
# ---------------------------------------------------------
class PerformanceTracker:
    def __init__(self, window_size=100):
        self.history = deque(maxlen=window_size)

    def add_metric(self, data: dict):
        self.history.append(data)

    def get_stats(self):
        if not self.history: return {}
        avg_latency = sum(d['latency'] for d in self.history) / len(self.history)
        avg_score = sum(d['eval_score'] for d in self.history) / len(self.history)
        total_retries = sum(d['retry_count'] for d in self.history)
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "avg_eval_score": round(avg_score, 2),
            "total_requests": len(self.history),
            "retry_rate": round(total_retries / len(self.history), 2)
        }


tracker = PerformanceTracker()


# ---------------------------------------------------------
# 3. 상태(State) 및 출력 스키마 정의
# ---------------------------------------------------------
class QuantState(TypedDict, total=False):
    # Input Data
    question: str

    news_context: str
    is_price_rising: bool
    market_iv: float

    # Process Data
    bull_opinion: str
    bear_opinion: str
    final_consensus: str

    market_trend: str
    risk_score: float
    news_sentiment: str
    divergence_note: str

    # 평가 & 개선
    eval_score: int
    eval_critique: str
    is_sufficient: bool
    retry_count: int

    # 성능 측정용
    start_time: float
    node_timings: Dict[str, float]

    # Final Outputs
    manager_view: str

    rag_context: str

    expected_returns: List[float]
    vol_vector: List[float]
    correlation_matrix: List[List[float]]
    covariance_matrix: List[List[float]]

    optimal_weights: List[float]


class JudgeOutput(BaseModel):
    final_consensus: str = Field(description="상승/하락 의견을 종합한 최종 합의문")
    market_trend: Literal["Bullish", "Bearish", "Volatile", "Neutral"] = Field(description="시장 방향성")
    risk_score: float = Field(ge=1.0, le=10.0, description="리스크 점수")
    news_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(description="뉴스 심리")
    divergence_note: str = Field(description="괴리 요약")


class EvalOutput(BaseModel):
    score: int = Field(description="답변의 정확성 및 관련성 점수 (1~10)")
    is_sufficient: bool = Field(description="추가 개선 없이 채택 가능한가")
    critique: str = Field(description="불충분하거나 보완이 필요한 부분에 대한 피드백")


# ---------------------------------------------------------
# 4. 노드 실행 시간 측정 데코레이터
# ---------------------------------------------------------
def measure_time(node_func):
    def wrapper(state: QuantState):
        start = time.perf_counter()
        result = node_func(state)
        end = time.perf_counter()

        # 타이밍 데이터 업데이트
        timings = state.get("node_timings", {})
        node_name = node_func.__name__
        timings[node_name] = round((end - start) * 1000, 2)  # ms 단위

        if result is None: result = {}
        result["node_timings"] = timings
        return result

    return wrapper


# ---------------------------------------------------------
# 4. LangGraph 노드 정의
# ---------------------------------------------------------

@measure_time
def debate_agent_node(state: QuantState):
    """프롬프트 엔지니어링 강화: 페르소나 및 데이터 기반 분석 강제"""
    news = state["news_context"]
    q = state["question"]
    retry_feedback = f"\n[이전 시도 피드백]: {state.get('eval_critique', '')}" if state.get('retry_count', 0) > 0 else ""

    system_msg = SystemMessage(content="당신은 전문 퀀트 트레이더입니다. 제공된 뉴스 지표와 매크로 상황을 정밀 분석하세요.")

    bull_p = f"상승론자 시각에서 분석하세요.{retry_feedback}\n뉴스: {news}\n질문: {q}"
    bear_p = f"하락론자 시각에서 분석하세요.{retry_feedback}\n뉴스: {news}\n질문: {q}"

    bull_op = llm.invoke([system_msg, HumanMessage(content=bull_p)]).content
    bear_op = llm.invoke([system_msg, HumanMessage(content=bear_p)]).content
    return {"bull_opinion": bull_op, "bear_opinion": bear_op, "retry_count": state.get("retry_count", 0)}

@measure_time
def judge_node(state: QuantState):
    """합의문 생성 및 Divergence 체크"""
    structured_llm = llm.with_structured_output(JudgeOutput)
    prompt = (
        f"CIO로서 Bull/Bear 의견을 분석하여 최종 합의문을 작성하세요.\n"
        f"Bull: {state['bull_opinion']}\n"
        f"Bear: {state['bear_opinion']}\n\n"
        f"중요: 현재 가격은 {'상승' if state['is_price_rising'] else '부진'} 중입니다. 이를 반영해 [Divergence]를 작성하세요."
    )
    res = structured_llm.invoke([HumanMessage(content=prompt)])
    return {
        "final_consensus": res.final_consensus,
        "market_trend": res.market_trend,
        "risk_score": res.risk_score,
        "news_sentiment": res.news_sentiment,
        "divergence_note": res.divergence_note
    }

@measure_time
def evaluator_node(state: QuantState):
    """ 생성된 답변의 정확성과 관련성 평가"""
    structured_eval = llm.with_structured_output(EvalOutput)

    prompt = (
        "당신은 금융 리포트 감사관입니다. 아래 생성된 결론이 원본 뉴스 상황과 질문에 얼마나 정확하게 부합하는지 평가하세요.\n\n"
        f"[원본 질문]: {state['question']}\n"
        f"[참고 뉴스]: {state['news_context']}\n"
        f"[생성된 결론]: {state['final_consensus']}\n\n"
        "평가 기준: 1. 직접적인 질문 답변 여부 2. 최신 지표 반영 여부 3. 논리적 모순성\n"
        "8점 미만이거나 개선이 필요하면 is_sufficient를 false로 설정하고 구체적인 비판(critique)을 작성하세요."
    )

    eval_res = structured_eval.invoke([HumanMessage(content=prompt)])

    # 최대 2회 재시도 제한 (무한 루프 방지)
    if state.get("retry_count", 0) >= 1:
        eval_res.is_sufficient = True

    return {
        "eval_score": eval_res.score,
        "is_sufficient": eval_res.is_sufficient,
        "eval_critique": eval_res.critique,
        "retry_count": state.get("retry_count", 0) + 1
    }

@measure_time
def scenario_rag_node(state: QuantState):
    """ 검색 전략 조정 - 합의문 기반 고도화 검색"""
    # 단순 질문 검색이 아닌, 평가를 통과한 '합의문'을 검색 키워드로 사용

    query = f"{state['market_trend']} {state['final_consensus'][:200]}"
    top = scenario_vs.similarity_search(query, k=1)

    if not top:
        return {"rag_context": "[참조 과거 시나리오 없음]"}

    md = top[0].metadata

    anchor_info = (
        f"\n[RAG Anchor: {md['name']}]\n- 설명: {md['desc']}\n"



        f"- 기준 mu: {md['mu']}\n- 기준 vol: {md['vol']}\n- 기준 corr: {md['corr']}"

    )

    # 최종 manager_view 생성
    price_action_text = "상승" if state["is_price_rising"] else "부진/하락"
    view = "📌 [AI Debate Market View]\n"
    view += f"- News Sentiment: {state['news_sentiment']}\n- Price Action: {price_action_text}\n\n"
    view += f"⚖️ [Judge Consensus]\n{state['final_consensus']}\n\n"
    view += f"[Divergence]\n- {state['divergence_note']}\n"
    view += f"[RAG Context]\n{anchor_info}\n"

    return {
        "rag_anchor_id": md.get("id"),

        "anchor_mu": md.get("mu"),
        "anchor_vol": md.get("vol"),
        "anchor_corr": md.get("corr"),
        "rag_context": anchor_info,
        "manager_view": view
    }

@measure_time
def quant_engine_node(state: QuantState):
    """LLM을 이용한 정량적 파라미터 미세 조정(Fine-tuning)"""

    prompt = (
        "SYSTEM: You are a quantitative risk management engine.\n"



        "당신은 금융 분석 전문가입니다. 아래 [현재 상황]과 RAG로 추출된 [참고 Anchor]을 결합 분석하여 "
        "5개 자산의 기대수익률(mu), 변동성(vol), 상관계수(corr)를 JSON으로 추정하세요.\n\n"

        f"현재 상황: {state['final_consensus']}\n"
        f"IV: {state['market_iv']}%\n"
        f"참고 Anchor: {state['rag_context']}\n\n"
        "반드시 유효한 JSON 형식으로만 응답하세요."

    )

    try:
        raw_response = llm.invoke([HumanMessage(content=prompt)]).content
        # JSON 파싱 로직 (간소화)
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        data = json.loads(raw_response[start:end])

        mu = data.get("mu", [0.01] * 5)
        vol = data.get("vol", [0.2] * 5)
        corr = np.array(data.get("corr", np.eye(5).tolist()))
        sigma = np.outer(vol, vol) * corr

        return {
            "expected_returns": mu,
            "vol_vector": vol,
            "correlation_matrix": corr.tolist(),
            "covariance_matrix": sigma.tolist()
        }

    except Exception as e:
        print(f"⚠️ Quant Engine Error: {e}")

        return {"expected_returns": [0.01] * 5}

@measure_time
def slsqp_optimizer_node(state: QuantState):
    """수학적 최적화로 최종 비중 산출"""
    mu = np.array(state.get("expected_returns", [0.01] * 5))

    def obj(w): return -np.dot(w, mu)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, 0.45)] * 5
    res = minimize(obj, [0.2] * 5, method='SLSQP', bounds=bounds, constraints=cons)
    return {"optimal_weights": res.x.tolist()}


# ---------------------------------------------------------
# 5. 워크플로우 구성 (평가 및 조건부 루프)
# ---------------------------------------------------------
def decide_refinement(state: QuantState):
    """평가 점수에 따라 재수행 여부 결정"""
    if state.get("is_sufficient"):
        return "approved"
    return "refine"


workflow = StateGraph(QuantState)

workflow.add_node("Debate", debate_agent_node)
workflow.add_node("Judge", judge_node)
workflow.add_node("Evaluate", evaluator_node)  # 평가 노드

workflow.add_node("ScenarioRAG", scenario_rag_node)
workflow.add_node("QuantEngine", quant_engine_node)

workflow.add_node("Optimizer", slsqp_optimizer_node)

workflow.set_entry_point("Debate")
workflow.add_edge("Debate", "Judge")
workflow.add_edge("Judge", "Evaluate")

# 조건부 엣지: 불충분하면 다시 토론으로, 충분하면 RAG로 진행
workflow.add_conditional_edges(
    "Evaluate",
    decide_refinement,
    {
        "refine": "Debate",
        "approved": "ScenarioRAG"
    }
)

workflow.add_edge("ScenarioRAG", "QuantEngine")
workflow.add_edge("QuantEngine", "Optimizer")
workflow.add_edge("Optimizer", END)

workflow_app = workflow.compile()

# ---------------------------------------------------------
# 6. FastAPI 서버 구축
# ---------------------------------------------------------
api_app = FastAPI(title="Test, Self-Improving Financial RAG API")


class AnalyzeRequest(BaseModel):
    question: str


@api_app.post("/analyze")
async def analyze_market(request: AnalyzeRequest):
    overall_start = time.perf_counter()

    # 가상의 뉴스 컨텍스트 및 지표 준비 (전처리)

    test_news = "2026-02-03 | 워시 연준 의장 후보 지명에 따른 금리 인하 기대감 혼조 및 환율 하락"

    try:
        initial_state = {
            "question": request.question,
            "news_context": test_news,
            "is_price_rising": True,
            "market_iv": 12.0,
            "retry_count": 0,
            "node_timings": {}
        }

        result = workflow_app.invoke(initial_state)

        overall_end = time.perf_counter()
        total_latency = (overall_end - overall_start) * 1000

        # 메트릭 저장
        tracker.add_metric({
            "latency": total_latency,
            "eval_score": result.get("eval_score", 0),
            "retry_count": result.get("retry_count", 0)
        })


        return {
            "status": "success",
            "performance": {
                "total_latency_ms": round(total_latency, 2),
                "node_breakdown": result.get("node_timings")
            },
            "evaluation": {
                "final_score": result.get("eval_score"),
                "total_attempts": result.get("retry_count"),
                "critique": result.get("eval_critique")
            },
            "manager_view": result.get("manager_view"),
            "quant_params": {
                "expected_returns": result.get("expected_returns"),

                "risk_score": result.get("risk_score")
            },
            "portfolio": {

                "weights": [round(w, 4) for w in (result.get("optimal_weights") or [])]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/metrics")
async def get_system_metrics():
    """시스템 전체 성능 통계를 반환."""
    stats = tracker.get_stats()
    if not stats:
        return {"message": "No data collected yet."}
    return stats


if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8088)
