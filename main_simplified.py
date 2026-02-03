import os
import json
import warnings
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# 수치 해석 및 데이터 수집
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import yfinance as yf

# --- 1. 환경 설정 및 LLM 초기화 ---
load_dotenv(override=True)
warnings.filterwarnings("ignore")

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


# --- 2. 상태 정의 및 스키마 ---
class QuantState(TypedDict, total=False):
    # 입력 및 기초 데이터
    question: str
    raw_news_data: List[Dict]
    news_context: str
    is_price_rising: bool
    macro_pred: Dict[str, Any]

    # 토론 데이터
    bull_opinion: str
    bear_opinion: str
    final_consensus: str

    # 가공된 핵심 필드 (요구사항)
    manager_view: str  # 최종 포맷팅된 뷰

    # 결과 데이터
    market_trend: str
    risk_score: float
    optimal_weights: List[float]
    final_report: str


class JudgeOutput(BaseModel):
    final_consensus: str = Field(description="최종 합의문")
    market_trend: Literal["Bullish", "Bearish", "Volatile", "Neutral"] = Field(description="시장 방향성")
    risk_score: float = Field(ge=1.0, le=10.0, description="리스크 점수")
    news_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(description="뉴스 심리")
    divergence_note: str = Field(description="괴리 요약")


# --- 3. 핵심 유틸리티 함수 (로직 준수) ---

def build_manager_view_from_debate(state: QuantState) -> str:
    """[Logic Rule] 토론 결과를 바탕으로 정교한 manager_view 문자열 구축"""
    price_action_text = "상승" if state['is_price_rising'] else "부진/하락"
    ns = state.get('news_sentiment', 'Neutral')
    dn = state.get('divergence_note', '괴리 없음')

    view = f"📌 [AI Debate Market View]\n"
    view += f"- News Sentiment: {ns}\n- Price Action: {price_action_text}\n\n"
    view += f"🟢 [Bull Case]\n{state['bull_opinion']}\n\n"
    view += f"🔴 [Bear Case]\n{state['bear_opinion']}\n\n"
    view += f"⚖️ [Judge Consensus]\n{state['final_consensus']}\n\n"
    view += f"[Divergence]\n- {dn}\n"
    view += "- NOTE: Price Action takes precedence over news sentiment.\n"
    return view


# --- 4. LangGraph 노드 구성 ---

def debate_agent_node(state: QuantState):
    """Bull/Bear 동시 분석 노드"""
    news = state['news_context']
    q = state['question']

    bull_p = f"금융 전략가로서 다음 뉴스({news})와 질문({q})에 대해 상승 시나리오를 논리적으로 서술하세요."
    bear_p = f"리스크 관리자로서 다음 뉴스({news})와 질문({q})에 대해 하락 리스크를 엄중히 경고하세요."

    bull_op = llm.invoke([HumanMessage(content=bull_p)]).content
    bear_op = llm.invoke([HumanMessage(content=bear_p)]).content

    return {"bull_opinion": bull_op, "bear_opinion": bear_op}


def judge_node(state: QuantState):
    """CIO가 토론을 결합하여 합의 도출"""
    structured_llm = llm.with_structured_output(JudgeOutput)
    prompt = (
        f"CIO로서 상승론({state['bull_opinion']})과 하락론({state['bear_opinion']})을 분석하여 "
        f"최종 합의문을 작성하세요. 한국어로 작성하되 기술 용어는 유지하세요."
    )
    res = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_consensus": res.final_consensus,
        "market_trend": res.market_trend,
        "risk_score": res.risk_score,
        "news_sentiment": res.news_sentiment,
        "divergence_note": res.divergence_note
    }


def view_generator_node(state: QuantState):
    """[핵심] 모든 정보를 취합하여 퀀트 엔진용 manager_view 생성"""
    m_view = build_manager_view_from_debate(state)
    return {"manager_view": m_view}


def slsqp_optimizer_node(state: QuantState):
    """수학적 최적화로 자산 비중 산출"""
    trend = state['market_trend'].lower()
    # 기대수익률 벡터 (Call_L, Call_S, Put_L, Put_S, Future)
    if trend == "bullish":
        mu = [0.08, -0.02, -0.06, 0.03, 0.12]
    elif trend == "bearish":
        mu = [-0.07, 0.03, 0.09, -0.02, -0.15]
    else:
        mu = [0.01, 0.01, 0.01, 0.01, 0.0]

    def obj(w):
        return -np.dot(w, mu)  # 수익 극대화

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, 0.45)] * 5

    res = minimize(obj, [0.2] * 5, method='SLSQP', bounds=bounds, constraints=cons)
    return {"optimal_weights": res.x.tolist()}


# --- 5. 워크플로우 정의 ---
workflow = StateGraph(QuantState)

workflow.add_node("Debate", debate_agent_node)
workflow.add_node("Judge", judge_node)
workflow.add_node("GenerateView", view_generator_node)
workflow.add_node("Optimizer", slsqp_optimizer_node)

workflow.set_entry_point("Debate")
workflow.add_edge("Debate", "Judge")
workflow.add_edge("Judge", "GenerateView")
workflow.add_edge("GenerateView", "Optimizer")
workflow.add_edge("Optimizer", END)

workflow_app = workflow.compile()

# --- 6. FastAPI 및 매크로 엔진 ---
api_app = FastAPI(title="OptiQ Manager View API")


def get_is_price_rising_mock():
    """매크로 Ridge 회귀를 모사한 가격 방향성 판단 (예시 상 True)"""
    return True


class AnalyzeRequest(BaseModel):
    question: str


@api_app.post("/analyze")
async def analyze_market(request: AnalyzeRequest): # Dict 대신 AnalyzeRequest 사용
    question = request.question

    # 1. 뉴스 데이터 준비 (예시 데이터 구조 활용)
    # 실제 구현 시 DB fetch_latest_news() 호출
    test_news = [
        {"date": "2026-02-03", "time": "20:41", "title": "워시 연준 의장 후보, 적극적 금리 인하 추진 관측"},
        {"date": "2026-02-03", "time": "16:35", "title": "코스피 사상 최고치 경신, 외국인 대거 순매수"}
    ]

    news_context = "\n".join([f"- {n['date']} | {n['title']}" for n in test_news])

    # 2. 초기 상태 설정
    initial_state: QuantState = {
        "question": question,
        "news_context": news_context,
        "is_price_rising": get_is_price_rising_mock(),
        "raw_news_data": test_news
    }

    # 3. 워크플로우 실행
    try:
        result = workflow_app.invoke(initial_state)

        return {
            "status": "success",
            "manager_view": result["manager_view"],  # 가공된 뷰 반환
            "quant_analysis": {
                "trend": result["market_trend"],
                "risk_score": result["risk_score"],
                "weights": [round(w, 4) for w in result["optimal_weights"]]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8088)
