import os
import json
import time
import warnings
import uvicorn
from datetime import datetime, timedelta, date
from typing import TypedDict, List, Dict, Optional, Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 금융 및 수치 계산
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import mysql.connector
from mysql.connector import pooling

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.stores import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- 1. 초기 설정 ---
load_dotenv(override=True)
warnings.filterwarnings("ignore")

# API 및 상수
MULTIPLIER = 50000
TOTAL_CAPITAL = 100000000

# LLM 설정 (로컬 서버 사용 기준)
llm = ChatOpenAI(
    model="local-llama",
    base_url="http://localhost:8090/v1",
    api_key="no-key-needed",
    temperature=0.0
)

embeddings = OpenAIEmbeddings(
    model="local-model",
    base_url="http://localhost:8090/v1",
    api_key="no-key-needed"
)


# --- 2. RAG 엔진 클래스 ---
class FinancialRAGEngine:
    def __init__(self):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.vectorstore = FAISS.from_documents(
            [Document(page_content="system init", metadata={"id": "init"})], embeddings
        )
        self.docstore = InMemoryStore()
        # 원본 코드의 CustomParentDocumentRetriever를 일반 FAISS 검색으로 간소화하여 예시 작성
        self.retriever = self.vectorstore.as_retriever()

    def rewrite_query(self, market_context: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "금융 퀀트 트레이더로서, 다음 상황과 유사한 과거 사례를 찾기 위한 최적화된 검색어를 작성하세요: {context}"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": market_context})

    def hybrid_search(self, query: str) -> str:
        optimized = self.rewrite_query(query)
        docs = self.retriever.get_relevant_documents(optimized)
        return "\n".join([d.page_content for d in docs])


rag_engine = FinancialRAGEngine()


# --- 3. 상태(State) 및 스키마 정의 ---
class QuantState(TypedDict, total=False):
    manager_view: str
    rag_context: str
    market_trend: str
    risk_aversion: float
    bull_opinion: str
    bear_opinion: str
    final_consensus: str
    console_report: str


class JudgeOutput(BaseModel):
    final_consensus: str = Field(description="최종 합의문")
    market_trend: Literal["Bullish", "Bearish", "Volatile", "Neutral"] = Field(description="시장 방향성")
    risk_score: float = Field(ge=1.0, le=10.0, description="리스크 점수")


# --- 4. LangGraph 노드 정의 ---

def rag_node(state: QuantState):
    context = rag_engine.hybrid_search(state['manager_view'])
    return {"rag_context": context}


def debate_node(state: QuantState):
    # Bull/Bear 토론 로직 (요약)
    bull_prompt = f"상승론자 입장에서 분석하세요: {state['manager_view']}"
    bear_prompt = f"하락론자 입장에서 분석하세요: {state['manager_view']}"

    bull_op = llm.invoke([HumanMessage(content=bull_prompt)]).content
    bear_op = llm.invoke([HumanMessage(content=bear_prompt)]).content

    return {"bull_opinion": bull_op, "bear_opinion": bear_op}


def judge_node(state: QuantState):
    structured_llm = llm.with_structured_output(JudgeOutput)
    prompt = f"토론 결과 요약: Bull:{state['bull_opinion']}, Bear:{state['bear_opinion']}. 최종 결론을 내리세요."
    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_consensus": result.final_consensus,
        "market_trend": result.market_trend,
        "risk_aversion": result.risk_score
    }


def quant_reporter_node(state: QuantState):
    # 실제 퀀트 계산 결과 리포트 생성 (가상 결과)
    report = f"### [Strategy Report]\n- Trend: {state['market_trend']}\n- Decision: {state['final_consensus']}"
    return {"console_report": report}


# --- 5. 그래프 구축 ---
workflow = StateGraph(QuantState)

workflow.add_node("RAG", rag_node)
workflow.add_node("Debate", debate_node)
workflow.add_node("Judge", judge_node)
workflow.add_node("Reporter", quant_reporter_node)

workflow.set_entry_point("RAG")
workflow.add_edge("RAG", "Debate")
workflow.add_edge("Debate", "Judge")
workflow.add_edge("Judge", "Reporter")
workflow.add_edge("Reporter", END)

workflow_app = workflow.compile()

# --- 6. FastAPI 서버 설정 ---
app = FastAPI(title="OptiQ Financial RAG API")


class QueryRequest(BaseModel):
    question: str


@app.post("/analyze")
async def analyze_market(request: QueryRequest):
    # LangGraph 실행
    inputs = {
        "manager_view": request.question
    }
    result = workflow_app.invoke(inputs)

    return {
        "trend": result.get("market_trend"),
        "consensus": result.get("final_consensus"),
        "report": result.get("console_report"),
        "rag_used": result.get("rag_context")[:200] + "..."  # RAG 참고 내용 일부 반환
    }


# 서버 직접 실행용
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)

