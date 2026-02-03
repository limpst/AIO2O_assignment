import os
import json
import warnings
import uvicorn
import numpy as np
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LangChain & RAG 관련 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- 금융 수치 및 최적화 ---
from scipy.optimize import minimize
from mysql.connector import pooling

# ---------------------------------------------------------
# 1. 설정 및 초기화 (R1: 데이터 수집 및 전처리 준비)
# ---------------------------------------------------------
load_dotenv(override=True)
warnings.filterwarnings("ignore")

# LLM 설정 (로컬 Llama 서버 혹은 OpenAI)
llm = ChatOpenAI(
    model="local-llama",
    base_url=os.getenv("LLM_SERVER_URL", "http://localhost:8090/v1"),
    api_key="no-key-needed",
    temperature=0.0,
    timeout=600
)

embeddings = OpenAIEmbeddings(
    model="local-model",
    base_url=os.getenv("LLM_SERVER_URL", "http://localhost:8090/v1"),
    api_key="no-key-needed"
)

# ---------------------------------------------------------
# 2. RAG 엔진 클래스 (R2, R3: FAISS 색인 및 검색 구현)
# ---------------------------------------------------------
class FinancialRAGEngine:
    def __init__(self):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        # 초기 더미 데이터로 벡터스토어 생성
        self.vectorstore = FAISS.from_documents(
            [Document(page_content="금융 시장 분석 시스템 가동", metadata={"source": "system"})],
            embeddings
        )

    def ingest_news(self, news_list: List[Dict]):
        """뉴스 데이터를 벡터 DB에 색인 (R2)"""
        docs = []
        for news in news_list:
            content = f"날짜: {news.get('date')}\n제목: {news.get('title')}\n본문: {news.get('body')}"
            docs.append(Document(page_content=content, metadata={"source": "news_db"}))
        
        if docs:
            self.vectorstore.add_documents(docs)
            print(f"✅ [RAG] {len(docs)}개 뉴스 색인 완료")

    def hybrid_search(self, query: str) -> str:
        """금융 특화 검색 (R3)"""
        # 쿼리 재작성: 모호한 질문을 금융 전문 용어로 변환
        rewrite_prompt = ChatPromptTemplate.from_template(
            "당신은 퀀트 분석가입니다. 다음 상황과 유사한 과거 사례를 찾기 위해 전문 용어(금리, 유동성, Volatility 등)를 "
            "포함한 검색 쿼리로 재작성하세요: {query}"
        )
        rewriter = rewrite_prompt | llm | StrOutputParser()
        optimized_query = rewriter.invoke({"query": query})
        
        # 벡터 검색
        docs = self.vectorstore.similarity_search(optimized_query, k=5)
        return "\n\n".join([d.page_content for d in docs])

rag_engine = FinancialRAGEngine()

# ---------------------------------------------------------
# 3. 분석 스키마 및 상태 정의
# ---------------------------------------------------------
class JudgeOutput(BaseModel):
    final_consensus: str = Field(description="상승/하락 의견을 종합한 최종 합의문")
    market_trend: Literal["Bullish", "Bearish", "Volatile", "Neutral"] = Field(description="최종 시장 방향성")
    risk_score: float = Field(ge=1.0, le=10.0, description="리스크 점수")
    divergence_note: str = Field(description="뉴스 심리와 가격 액션 간의 괴리 분석")

class QuantState(TypedDict, total=False):
    question: str
    rag_context: str
    bull_opinion: str
    bear_opinion: str
    final_consensus: str
    market_trend: str
    risk_score: float
    is_price_rising: bool
    optimal_weights: List[float]

# ---------------------------------------------------------
# 4. 워크플로우 노드 (R4, R5: 답변 생성 및 평가/개선)
# ---------------------------------------------------------
def retrieval_node(state: QuantState):
    context = rag_engine.hybrid_search(state['question'])
    return {"rag_context": context}

def debate_node(state: QuantState):
    """Bull vs Bear 토론을 통한 답변 생성 (R4)"""
    ctx = state['rag_context']
    q = state['question']
    
    bull_p = f"금융 분석가로서 다음 정보({ctx})를 바탕으로 질문({q})에 대해 가장 낙관적인 상승 시나리오를 제시하세요."
    bear_p = f"리스크 관리자로서 다음 정보({ctx})를 바탕으로 질문({q})에 대해 가장 보수적인 하락 리스크를 경고하세요."
    
    bull_op = llm.invoke([HumanMessage(content=bull_p)]).content
    bear_op = llm.invoke([HumanMessage(content=bear_p)]).content
    
    return {"bull_opinion": bull_op, "bear_opinion": bear_op}

def judge_node(state: QuantState):
    """CIO의 최종 판단 및 가드레일 적용 (R5)"""
    structured_llm = llm.with_structured_output(JudgeOutput)
    
    # 평가 루프: 실제 가격 방향성(Price Action) 가중치 부여
    prompt = (
        f"당신은 CIO입니다. 상승론({state['bull_opinion']})과 하락론({state['bear_opinion']})을 결합 분석하세요.\n"
        f"중요: 현재 실제 가격 지표는 {'상승' if state['is_price_rising'] else '하락/보합'} 중입니다.\n"
        f"만약 뉴스가 부정적임에도 가격이 오르고 있다면 [Divergence] 섹션에 'Bullish Climber'를 명시하세요."
    )
    
    res = structured_llm.invoke([HumanMessage(content=prompt)])
    return {
        "final_consensus": res.final_consensus,
        "market_trend": res.market_trend,
        "risk_score": res.risk_score,
        "divergence_note": res.divergence_note
    }

def optimizer_node(state: QuantState):
    """수학적 포트폴리오 최적화 (금융 모델링 경험 활용)"""
    trend = state['market_trend'].lower()
    # 기대수익률(mu) 매핑
    if trend == "bullish": mu = [0.1, -0.02, -0.05, 0.03, 0.15]
    elif trend == "bearish": mu = [-0.1, 0.03, 0.08, -0.02, -0.15]
    else: mu = [0.01] * 5

    def objective(w): return -np.dot(w, mu)  # 기대 수익 극대화
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = [(0, 0.5)] * 5 # 단일 자산 비중 50% 제한
    
    res = minimize(objective, [0.2]*5, method='SLSQP', bounds=bounds, constraints=cons)
    return {"optimal_weights": res.x.tolist()}

# ---------------------------------------------------------
# 5. 워크플로우 컴파일
# ---------------------------------------------------------
workflow = StateGraph(QuantState)
workflow.add_node("Retrieve", retrieval_node)
workflow.add_node("Debate", debate_node)
workflow.add_node("Judge", judge_node)
workflow.add_node("Optimize", optimizer_node)

workflow.set_entry_point("Retrieve")
workflow.add_edge("Retrieve", "Debate")
workflow.add_edge("Debate", "Judge")
workflow.add_edge("Judge", "Optimize")
workflow.add_edge("Optimize", END)

rag_app = workflow.compile()

# ---------------------------------------------------------
# 6. REST API 서버 구축 (R6: FastAPI 활용)
# ---------------------------------------------------------
api_app = FastAPI(title="금융 지능형 RAG API")

class AnalysisRequest(BaseModel):
    question: str
    is_price_rising: Optional[bool] = True

@api_app.post("/analyze")
async def analyze_finance(request: AnalysisRequest):
    """외부 호출 가능한 RAG 분석 엔드포인트 (R6)"""
    start_time = datetime.now() # (R7: 성능 측정용)
    
    try:
        inputs = {
            "question": request.question,
            "is_price_rising": request.is_price_rising
        }
        result = rag_app.invoke(inputs)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "execution_time_sec": execution_time,
            "analysis": {
                "trend": result.get("market_trend"),
                "risk_score": result.get("risk_score"),
                "consensus": result.get("final_consensus"),
                "divergence": result.get("divergence_note")
            },
            "portfolio": {
                "asset_weights": result.get("optimal_weights")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 7. 초기 실행 데이터 (R1: 데이터 로드 예시)
# ---------------------------------------------------------
def seed_data():
    sample_news = [
        {"date": "2026-02-03", "title": "연준, 금리 인하 속도 조절 시사", "body": "매파적 위원들의 발언으로 금리 인하 횟수가 줄어들 가능성..."},
        {"date": "2026-02-03", "title": "코스피 외국인 대거 순매수", "body": "실적 개선 기대감에 삼성전자, 하이닉스 위주 외국인 자금 유입..."}
    ]
    rag_engine.ingest_news(sample_news)

if __name__ == "__main__":
    seed_data()
    uvicorn.run(api_app, host="0.0.0.0", port=8088)
