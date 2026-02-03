import time
import unicodedata
import websocket
import json
import requests
import re
import os
import mysql.connector
import threading
import queue
import multiprocessing
import urllib3  # 추가
import glob
import pdfplumber   # PDF 처리를 위한 라이브러리
import shutil       # 파일 이동을 위한 모듈
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mysql.connector import pooling
from html import unescape
from pathlib import Path
from requests.exceptions import HTTPError
from collections import deque

# [추가] SSL 경고 메시지 숨기기 (로그 깔끔하게)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. 환경 변수 및 기본 설정
# ==========================================

load_dotenv()

n_cpu_cores = multiprocessing.cpu_count()

# 필수 값 확인
if not os.getenv("DB_PASSWORD"):  # not os.getenv("LS_ACCESS_TOKEN") or
    # print("❌ [Error] .env 파일이 없거나 필수 환경 변수(TOKEN, PASSWORD)가 누락되었습니다.")
    print("❌ [Error] .env 파일이 없거나 DB_PASSWORD가 누락되었습니다.")
    exit(1)

# ==========================================
# 1. 설정 및 상수 정의 (Configuration)
# ==========================================

WS_URL = "wss://openapi.ls-sec.co.kr:9443/websocket"
API_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
# ACCESS_TOKEN = os.getenv("LS_ACCESS_TOKEN", "DUMMY_TOKEN")  # PDF 모드일 경우 토큰 없어도 에러 안 나게 처리
ACCESS_TOKEN = None  # 런타임 발급/갱신

# REST endpoint 상수 (중복 제거)
# MARKET_DATA_EP = "/futureoption/market-data"
STOCK_INVESTINFO_EP = "/stock/investinfo"

# ==========================================
# [PATCH] LS OAuth2 Token Manager + Safe REST Wrapper
# ==========================================
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")

TOKEN_URL = f"{API_BASE_URL}/oauth2/token"

_ACCESS_TOKEN = None
_ACCESS_TOKEN_EXPIRES_AT = 0  # epoch seconds

# [중요] 로컬 Llama 서버 주소 설정
LLM_SERVER_URL = "http://localhost:8090/v1"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "LLM")
}

# PDF 파일 경로 설정
PDF_BASE_PATH = Path(r"v:\Users\leeli\PycharmProjects\PythonProject1\data\pdf")

# 시스템 날짜를 읽어와 실제 감시할 경로 생성 (e.g. .../data/pdf/20251228)
current_date_str = datetime.now().strftime("%Y%m%d")
PDF_DIR_PATH = PDF_BASE_PATH / current_date_str

# news_queue = queue.Queue()
# 큐 사이즈 제한 설정 (메모리 폭주 방지)
# 뉴스가 너무 많이 쌓이면 워커가 처리할 수 없으므로 제한을 둡니다.
MAX_QUEUE_SIZE = 30
news_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# DB Pool 생성
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="news_pool",
        pool_size=3,
        pool_reset_session=True,
        **DB_CONFIG
    )
    print("✅ [System] DB Connection Pool 생성 완료")
except Exception as e:
    print(f"❌ [System] DB Pool 생성 실패: {e}")
    exit(1)

# ==========================================
# [PATCH] Context-safe 요약 유틸 (8k/16k에서도 안 터지게)
# - 토크나이저 없이도 안전하게 동작하도록 "문자 길이" 기준으로 상한을 둠
# - 최종 요약 입력을 항상 제한 + 단계적(계층) 축약
# ==========================================

# 대략적인 토큰 추정(한국어/영어 혼합 환경에서 안전하게 잡기 위해 보수적으로)
# 경험적으로 "1 토큰 ~= 3~4 chars"가 많지만, 안전하게 3 chars/token로 가정
_CHARS_PER_TOKEN_EST = 3

# 서버 컨텍스트(n_ctx)에 따라 조정 (8k/16k 둘 다 안전하게)
# - 8k 기준: 입력(프롬프트) 16k chars 정도면 대체로 안전
# - 16k 기준: 입력 32k chars 정도 가능
# 여기서는 "절대 안 터지게"가 목표이므로 보수적으로 낮춤
LLM_MAX_USER_CHARS = 12000      # user_prompt 최대 길이(하드 컷)
LLM_MAX_COMBINED_CHARS = 9000  # chunk summary 합친 combined 최대 길이
LLM_MAX_CHUNK_SUMMARIES = 14    # 1차 청크 요약 최대 개수(너무 많으면 계층 축약)

def _safe_truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.88)]
    tail = text[-int(max_chars * 0.07):] if max_chars >= 600 else ""
    return (head + "\n...\n" + tail).strip()

def _build_user_prompt(prefix: str, content: str, max_chars: int = LLM_MAX_USER_CHARS) -> str:
    return _safe_truncate(f"{prefix}\n{content}".strip(), max_chars)

def call_llm_api(system_prompt, user_prompt, max_tokens=2048):
    """
    OpenAI 스타일 API 호출 헬퍼 함수 (컨텍스트 초과 방지)
    - user_prompt 하드 컷
    - 컨텍스트 초과 에러면 더 줄여 1회 재시도
    """
    user_prompt = _safe_truncate(user_prompt or "", LLM_MAX_USER_CHARS)

    try:
        with llm_lock:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=False,
                timeout=300
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        msg = str(e)
        print(f"⚠️ [LLM API Error] 서버 통신 실패: {e}")
        print(f"   - user_prompt length: {len(user_prompt) if user_prompt else 0}")

        if ("exceeds the available context size" in msg) or ("exceed_context_size" in msg) or ("n_ctx" in msg):
            shorter = _safe_truncate(user_prompt, 6000)
            print(f"🔧 [LLM Retry] 컨텍스트 초과 -> prompt 축소 후 재시도 ({len(shorter)} chars)")
            try:
                with llm_lock:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": shorter}
                        ],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        stream=False,
                        timeout=120
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e2:
                print(f"⚠️ [LLM API Error] 재시도도 실패: {e2}")
                return ""

        return ""

# ==========================================
# 2. LLM 클라이언트 초기화 (OpenAI 호환)
# ==========================================

print(f"⏳ [System] 로컬 LLM 서버({LLM_SERVER_URL})에 연결 설정 중...")

try:
    # OpenAI 클라이언트 인스턴스 생성
    client = OpenAI(
        base_url=LLM_SERVER_URL,
        api_key="no-key-needed"  # 로컬 서버라 키는 아무거나 입력
    )
    print("✅ [System] LLM 클라이언트 준비 완료")
except Exception as e:
    print(f"⚠️ [Warning] 클라이언트 설정 중 에러: {e}")

# LLM 호출 직렬화를 위한 전역 락 (서버 과부하 방지용)
llm_lock = threading.Lock()

# 키워드 정의
from keywords import (
    MACRO_KEYWORDS,             # MACRO_KEYWORDS = ["금리", "환율", "CPI", "PPI", "FOMC", "연준", "GDP", "물가", "유가"]
    GLOBAL_KEYWORDS,            # GLOBAL_KEYWORDS = ["나스닥", "다우", "S&P500", "뉴욕증시", "테슬라", "엔비디아", "애플", "TSMC"]
    DOMESTIC_MARKET_KEYWORDS,   # DOMESTIC_MARKET_KEYWORDS = ["코스피", "코스닥", "외국인", "기관", "순매수", "공매도"]
    ETC_KEYWORDS,               # ETC_KEYWORDS = ["비트코인", "가상화폐", "IPO", "공모주"]
)

def issue_ls_access_token(force: bool = False) -> str:
    """
    LS OAuth2 Client Credentials 토큰 발급/갱신
    - force=True: 무조건 재발급
    """
    global _ACCESS_TOKEN, _ACCESS_TOKEN_EXPIRES_AT

    now = int(time.time())

    # 만료 60초 전이면 미리 갱신
    if (not force) and _ACCESS_TOKEN and now < (_ACCESS_TOKEN_EXPIRES_AT - 60):
        return _ACCESS_TOKEN

    if not APP_KEY or not APP_SECRET:
        raise RuntimeError("❌ APP_KEY/APP_SECRET(.env) 누락")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecretkey": APP_SECRET,
        "scope": "oob",
    }

    resp = requests.post(TOKEN_URL, headers=headers, data=data, verify=False, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(f"❌ [LS] 토큰 발급 실패: {resp.status_code} / {resp.text}")

    j = resp.json()
    token = j.get("access_token")
    expires_in = int(j.get("expires_in", 1800))  # 없으면 30분 가정

    if not token:
        raise RuntimeError(f"❌ [LS] 토큰 응답에 access_token 없음: {j}")

    _ACCESS_TOKEN = token
    _ACCESS_TOKEN_EXPIRES_AT = now + expires_in
    print("✅ [LS] Access Token 발급/갱신 완료")

    return _ACCESS_TOKEN


def get_headers(tr_cd, tr_cont="N"):
    token = issue_ls_access_token(force=False)
    return {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {token}" if token else "Bearer ",
        "tr_cd": tr_cd,
        "tr_cont": tr_cont,
        "mac_address": "00:11:22:33:44:55"
    }


def ls_post(endpoint_path: str, tr_cd: str, body: dict, timeout: int = 5) -> dict:
    """
    LS REST 공통 POST
    - 401이면 토큰 강제 재발급 후 1회 재시도
    """
    url = f"{API_BASE_URL}{endpoint_path}"

    headers = get_headers(tr_cd)
    resp = requests.post(url, headers=headers, data=json.dumps(body), verify=False, timeout=timeout)

    if resp.status_code == 401:
        issue_ls_access_token(force=True)
        headers = get_headers(tr_cd)
        resp = requests.post(url, headers=headers, data=json.dumps(body), verify=False, timeout=timeout)

    resp.raise_for_status()
    return resp.json()

# ==========================================
# 3. 유틸리티 함수 (정제, 병합, DB)
# ==========================================

def get_safe_pdf_path(base_dir, date_str, file_name):
    """
    파일 경로를 안전하게 생성하고 존재 여부를 체크하는 함수
    """
    # 1. Path 객체를 사용하여 OS별 경로 구분자(/ vs \) 이슈 자동 해결
    # 공백이나 한글이 포함된 파일명도 안전하게 처리합니다.
    base_path = Path(base_dir)
    full_path = base_path / "data" / "pdf" / date_str / file_name

    # 2. 상위 디렉토리가 존재하는지 먼저 확인
    if not full_path.parent.exists():
        print(f"⚠️ [System] 디렉토리 경로가 존재하지 않습니다: {full_path.parent}")
        # 필요 시 디렉토리 생성 로직 추가
        # full_path.parent.mkdir(parents=True, exist_ok=True)
        return None

    # 3. 파일 자체가 존재하는지 확인
    if not full_path.exists():
        print(f"❌ [Errno 2] 파일을 찾을 수 없습니다: {full_path}")
        return None

    return full_path

def clean_financial_text(text: str) -> str:
    """금융 텍스트 정제"""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'^[A-Za-z0-9]+OutBlock\d+\s+', '', text)
    lines = text.splitlines()
    merged_lines = []

    bullet_pattern = re.compile(r'^[\*\-•※\[]')
    finance_symbol_pattern = re.compile(r'^[▲▼△▽↑↓]')
    starts_with_number = re.compile(r'^[▲▼△▽↑↓]\s*[0-9\.]')

    for line in lines:
        line = line.strip()
        if not line: continue

        if not merged_lines:
            merged_lines.append(line)
            continue
        prev_line = merged_lines[-1]

        if bullet_pattern.match(line):
            merged_lines.append(line)
        elif finance_symbol_pattern.match(line):
            if starts_with_number.match(line):
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
        elif prev_line.endswith('.') or prev_line.endswith(':'):
            merged_lines.append(line)
        else:
            merged_lines[-1] += " " + line

    result = "\n".join(merged_lines)
    result = re.sub(r'(\n)([▲▼△▽↑↓])(?!\s*[0-9])', r'\n\n\2', result)
    return result


def clean_base_text(text: str) -> str:
    """HTML 및 노이즈 제거"""
    if not text or not isinstance(text, str):
        return ""
    text = unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'(@media.*?\{.*?\})|(\{.*?\})', '', text, flags=re.DOTALL)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'^[A-Za-z0-9]+OutBlock\d+\s+', '', text)
    text = re.sub(r'[^\w\s.,\'"()%\+/\-▲▼△▽↑↓]', '', text)
    return text


def merge_news_bodies(news_bodies):
    """뉴스 본문 배열 병합"""
    merged_lines = []
    for news in news_bodies:
        line = news['sBody'].strip()
        if not line:
            continue
        if not merged_lines:
            merged_lines.append(line)
            continue
        if merged_lines[-1].endswith('.') or merged_lines[-1].endswith(':'):
            merged_lines.append(line)
        else:
            merged_lines[-1] += " " + line
    return "\n".join(merged_lines)


def insert_to_db(data):
    """MySQL 데이터 저장"""
    conn = None
    try:
        conn = db_pool.get_connection()
        # [추가] 연결이 유효한지 확인하고 필요하면 재연결
        if not conn.is_connected():
            conn.reconnect(attempts=3, delay=1)

        if conn.is_connected():
            cursor = conn.cursor()
            # realkey가 PDF인 경우 파일명 등으로 대체됨
            insert_query = """
                           INSERT INTO news_data (date, time, id, realkey, title, bodysize, category, body)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                           """
            cursor.execute(insert_query, data)
            conn.commit()
    except mysql.connector.Error as err:
        print(f"❌ DB 에러: {err}")
    finally:
        if conn:
            conn.close()


# ==========================================
# 4. 키워드 기반 1차 분류
# ==========================================

def quick_keyword_classify(title: str) -> str | None:
    t = title.strip()
    if any(k in t for k in MACRO_KEYWORDS): return "거시경제"

    if any(k in t for k in DOMESTIC_MARKET_KEYWORDS): return "국내 시황"

    if any(k in t for k in GLOBAL_KEYWORDS): return "해외 증시"

    if any(k in t for k in ETC_KEYWORDS): return "기타"

    return None


# ==========================================
# 5. 요약 + 분류 통합 LLM 호출 (OpenAI API 방식)
# ==========================================
#
# def call_llm_api(system_prompt, user_prompt, max_tokens=1024):
#     """
#     OpenAI 스타일 API 호출 헬퍼 함수
#     """
#     try:
#         # 스레드 락 사용 (서버에 동시 요청이 너무 몰리지 않도록 조절)
#         with llm_lock:
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",  # 로컬 서버에서는 모델명 무시됨 (서버에 로드된 모델 사용)
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=max_tokens,
#                 stream=False,  # 스트리밍 끔 (한 번에 받기)
#                 timeout = 120  # 60초 지나면 에러 발생시키고 다음으로 넘어감
#             )
#             return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"⚠️ [LLM API Error] 서버 통신 실패: {e}")
#         print(f"   - user_prompt length: {len(user_prompt) if user_prompt else 0}")
#         return ""

def get_heuristic_summary(text: str, max_sentences: int = 3) -> str:
    """LLM 실패 시 작동하는 규칙 기반 요약 (첫 문장과 마지막 문장 추출)"""
    if not text: return ""
    # 문장 단위 분리 (단순 마침표 기준)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= max_sentences:
        return text

    # 앞 2문장 + 뒤 1문장 조합
    head = " ".join(sentences[:2])
    tail = sentences[-1]
    return f"[자동추출] {head} ... {tail}"

def summarize_and_classify(text: str, title: str) -> tuple[str, str]:
    """
    1) 긴 텍스트는 청크 단위 요약 후 최종 요약
    2) 최종 요약 + 제목을 이용해 LLM으로 카테고리 분류
    """
    category = "기타_LLM"
    final_summary = "내용 없음"

    if not text:
        return "내용 없음", "기타"

    # 1차: 키워드 분류
    kw_category = quick_keyword_classify(title)

    # if kw_category in ["거시경제"]:
    # 텍스트 분할 (LangChain Splitter 활용)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)

    summary_system_prompt = (
        "역할: 당신은 투자자 관점의 전문 요약가이다.\n"
        "목표: 사용자가 제공한 뉴스/기사/리포트/공지 텍스트에서 투자자가 반드시 알아야 할 핵심 정보만 추출해 요약한다.\n\n"
        "출력 규칙:\n"
        "- 한국어로 작성한다.\n"
        "- 제목, 서론, 인사말, 목록 표기 없이 본문으로 바로 시작한다.\n"
        "- 정확히 3~5문장으로만 작성한다.\n"
        "- 아래 4가지 요소를 가능한 한 포함한다:\n"
        "  1) 중요한 수치/통계(금액, 성장률, 매출/이익, 물가, 금리, 점유율 등)\n"
        "  2) 관련 기업명/기관/인물\n"
        "  3) 주요 사건과 시장 동향(규제, 인수합병, 실적, 정책, 수요/공급 변화 등)\n"
        "  4) 경제적 영향 및 향후 전망(텍스트에 근거한 범위에서만)\n\n"
        "제약:\n"
        "- 불필요한 배경 설명, 감정적 표현, 과도한 세부사항은 제외한다.\n"
        "- 원문에 없는 내용은 추측하거나 단정하지 않는다.\n"
        "- 데이터가 불명확하면 '원문에 수치/정보 없음'처럼 간단히 명시한다.\n"
    )

    final_system_prompt = (
        "역할: 당신은 투자자 관점의 최종 요약 편집자이다.\n"
        "목표: 아래 '부분 요약들'을 읽고, 중복을 제거한 뒤 핵심만 남겨 최종 요약을 만든다.\n"
        "출력 규칙:\n"
        "- 한국어로 작성한다.\n"
        "- 정확히 3~5문장.\n"
        "- 숫자/기업/기관/정책/시장영향을 우선한다.\n"
        "- 원문에 없는 내용은 추가하지 않는다.\n"
    )

    def _llm_summary_single(chunk_text: str) -> str:
        # 새로 만든 API 호출 함수 사용
        # result = call_llm_api(summary_system_prompt, f"뉴스 본문:\n{chunk_text}")
        user_prompt = _build_user_prompt("뉴스 본문:", chunk_text, max_chars=LLM_MAX_USER_CHARS)
        result = call_llm_api(summary_system_prompt, user_prompt, max_tokens=2048)

        # 불필요한 서두 제거
        result = re.sub(r"^(\s*요약\s*[:\-\]]?|.*?요약해\s*드리겠습니다[.]?)", "", result).strip()
        return result

    def _llm_final(partials: str) -> str:
        partials = _safe_truncate(partials, LLM_MAX_COMBINED_CHARS)
        user_prompt = _build_user_prompt("부분 요약들:", partials, max_chars=LLM_MAX_USER_CHARS)
        result = call_llm_api(final_system_prompt, user_prompt, max_tokens=2048)
        result = re.sub(r"^(\s*요약\s*[:\-\]]?|.*?요약해\s*드리겠습니다[.]?)", "", result).strip()
        return result

    def _hier_reduce(summaries: list[str]) -> str:
        summaries = [s for s in summaries if s and s.strip()]
        if not summaries:
            return ""

        q = deque(summaries)
        BATCH_SIZE = 5

        while True:
            if len(q) == 1:
                return q[0]

            reduced = []
            while q:
                batch = []
                while q and len(batch) < BATCH_SIZE:
                    batch.append(q.popleft())
                out = _llm_final("\n".join(batch))
                if out:
                    reduced.append(out)

            if not reduced:
                return ""
            q = deque(reduced)

    # 1) 청크 요약
    if len(chunks) == 1:
        print(f"🧩 [System] 텍스트 1개의 청크로 처리합니다.")
        # final_summary = _llm_summary_single(text)
        final_summary = _llm_summary_single(_safe_truncate(text, LLM_MAX_USER_CHARS))
    else:
        print(f"🧩 [System] 긴 텍스트 분할 처리 ({len(chunks)}개)")
        chunk_summaries = []
        # for i, chunk in enumerate(chunks):
        #     print(f"   ... {i + 1}/{len(chunks)} 번째 청크 요약 중")
        # 청크가 너무 많으면 앞/뒤 일부만 1차 요약 (컨텍스트/시간 보호)
        if len(chunks) > LLM_MAX_CHUNK_SUMMARIES:
            head_n = LLM_MAX_CHUNK_SUMMARIES // 2
            tail_n = LLM_MAX_CHUNK_SUMMARIES - head_n
            selected = chunks[:head_n] + chunks[-tail_n:]
            print(f"🧯 [SAFE] 청크 과다({len(chunks)}개) -> {len(selected)}개(앞{head_n}+뒤{tail_n})만 1차 요약")
        else:
            selected = chunks

        for i, chunk in enumerate(selected):
            print(f"   ... {i + 1}/{len(selected)} 번째 청크 요약 중")
            summary = _llm_summary_single(chunk)
            chunk_summaries.append(summary)

        # combined = "\n".join(chunk_summaries)
        #
        # if len(combined) > 1000:
        #     print("🏁 [System] 최종 요약본 생성 중...")
        #     final_summary = _llm_summary_single(combined)
        # else:
        #     print("🏁 [System] 최종 요약본 생성 ...")
        #     final_summary = combined
        combined = "\n".join([s for s in chunk_summaries if s and s.strip()])
        combined = _safe_truncate(combined, LLM_MAX_COMBINED_CHARS)

        # 최종 요약(컨텍스트 안전)
        print("🏁 [System] 최종 요약본 생성 중... (n_ctx=8192 안전 모드)")
        if len(chunk_summaries) > 5:
            final_summary = _hier_reduce(chunk_summaries)
        else:
            final_summary = _llm_final(combined)

        if not final_summary:
            final_summary = _safe_truncate(text, 1200)

    # 2) 카테고리 분류
    if kw_category is not None:
        category = kw_category + "_KEY"
    else:
        classify_system = (
            "역할: 당신은 금융 뉴스 분류기이다.\n"
            "임무: 사용자가 제공한 뉴스 텍스트를 아래 5개 카테고리 중 하나로만 분류한다.\n\n"
            "카테고리 정의(택1):\n"
            "1) 거시경제: 금리, 환율, 유가, CPI/PPI, 고용/성장 지표, 중앙은행(연준/Fed 등), 경기/인플레이션.\n"
            "2) 해외 증시: 미국/해외 주가지수, 해외 시장 동향, 해외 상장기업(예: 애플, 엔비디아 등) 관련.\n"
            "3) 국내 시황: 코스피/코스닥 지수, 국내 증시 전반, 외국인/기관/개인 수급, 프로그램 매매.\n"
            "4) 주도 섹터: 국내 개별 기업(실적/공시/수주/합병 등) 또는 국내 산업/테마(반도체, 2차전지, 바이오 등).\n"
            "5) 기타: 가상자산, 정책/규제, IPO/상장, 지정학/전쟁/제재, 원자재 이슈 중 위 1~4에 명확히 안 맞는 경우.\n\n"
            "출력 규칙:\n"
            "- 반드시 아래 5개 라벨 중 하나만 '그대로' 출력한다: 거시경제, 해외 증시, 국내 시황, 주도 섹터, 기타\n"
            "- 추가 설명, 부연, 기호, 따옴표, 줄바꿈 없이 라벨만 출력한다.\n"
        )
        safe_title = _safe_truncate(title or "", 800)
        safe_summary = _safe_truncate(final_summary or "", 2500)

        classify_user = (
            f"다음 뉴스(제목+요약)를 읽고, 카테고리 라벨 1개만 선택해 출력하세요.\n"
            f"출력은 라벨만(추가 설명 금지): 거시경제, 해외 증시, 국내 시황, 주도 섹터, 기타\n\n"
            f"우선순위 규칙:\n"
            f"- 1순위: [뉴스 요약] 내용을 가장 높은 가중치로 판단한다.\n"
            f"- 2순위: 요약이 모호하거나 정보가 부족할 때만 [뉴스 제목]을 참고한다.\n"
            f"- 제목과 요약이 충돌하면 요약을 따른다.\n\n"
            f"[뉴스 제목]\n{safe_title}\n\n"
            f"[뉴스 요약]\n{safe_summary}\n\n"
            f"카테고리:"
        )
        classify_user = _safe_truncate(classify_user, LLM_MAX_USER_CHARS)

        # 분류 요청
        result = call_llm_api(classify_system, classify_user, max_tokens=50)

        valid_categories = ["거시경제", "해외 증시", "국내 시황", "주도 섹터", "기타"]
        for cat in valid_categories:
            if cat in result:
                category = cat + "_LLM"
                break

    return final_summary, category


# ==========================================
# 6. 뉴스 본문 조회 및 구조화
# ==========================================

def refine_financial_structure(text: str) -> str:
    """끊어진 문장 연결 및 금융 기호 구조화"""
    lines = text.splitlines()
    merged_lines = []
    bullet_pattern = re.compile(r'^[\*\-•※\[]')
    finance_symbol_pattern = re.compile(r'^[▲▼△▽↑↓]')
    starts_with_number = re.compile(r'^[▲▼△▽↑↓]\s*[0-9\.]')

    for line in lines:
        line = line.strip()
        if not line: continue
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev_line = merged_lines[-1]

        if bullet_pattern.match(line):
            merged_lines.append(line)
        elif finance_symbol_pattern.match(line):
            if starts_with_number.match(line):
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
        elif prev_line.endswith('.') or prev_line.endswith(':'):
            merged_lines.append(line)
        else:
            merged_lines[-1] += " " + line

    result = "\n".join(merged_lines)
    result = re.sub(r'(\n)([▲▼△▽↑↓])(?!\s*[0-9])', r'\n\n\2', result)
    result = re.sub(r'[ \t]+', ' ', result)
    return result


# def get_headers(tr_cd, tr_cont="N"):
#     return {
#         "Content-Type": "application/json; charset=UTF-8",
#         "Authorization": f"Bearer {ACCESS_TOKEN}",
#         "tr_cd": tr_cd,
#         "tr_cont": tr_cont,
#         "mac_address": "00:11:22:33:44:55"
#     }


def fetch_news_body(news_id):
    """REST API를 통해 뉴스 상세 본문 조회"""
    # news_id(realkey)가 없으면 요청하지 않음
    if not news_id:
        print("⚠️ [Skip] news_id(realkey)가 비어있습니다.")
        return None

    # 공백 제거 등 전처리
    clean_id = str(news_id).strip()

    data = {"t3102InBlock": {"sNewsno": clean_id}}

    max_retries = 3   # 최대 3번 재시도

    for attempt in range(max_retries):

        try:
            # [REFAC] 공통 REST Wrapper 사용 (401 재시도 일원화)
            response_json = ls_post(STOCK_INVESTINFO_EP, "t3102", data, timeout=5)

            # 응답 내에 에러 메시지가 있는지 확인 (API 특성에 따라 200 OK라도 내부에러일 수 있음)

            if "rsp_cd" in response_json and response_json["rsp_cd"] != "00000":
                print(f"⚠️ API 응답 에러: {response_json.get('rsp_msg', '알 수 없는 에러')}")
                return None

            if "t3102OutBlock1" not in response_json:
                return None

            news_body = response_json["t3102OutBlock1"]
            joined_body = merge_news_bodies(news_body)
            cleaned_body = clean_base_text(joined_body)
            refined_body = refine_financial_structure(cleaned_body)

            return refined_body

        except requests.exceptions.Timeout:
            print(f"⏰ 요청 시간 초과 (시도 {attempt + 1}/{max_retries})")
            time.sleep(1)
        except HTTPError as e:
            # raise_for_status()에서 발생한 HTTP 에러 처리
            status = getattr(e.response, "status_code", None)

            if status == 500:
                print(f"⚠️ [500 Error] 서버 내부 오류 (시도 {attempt + 1}/{max_retries}) - ID: {clean_id}")
                time.sleep(1)
                continue
            elif status == 401:
                # ls_post 내부에서 1회 재시도까지 수행하지만,
                # 혹시 모를 케이스를 위해 로그만 남기고 다음 루프로 넘김
                print(f"🔑 [401] 인증 오류 지속 (시도 {attempt + 1}/{max_retries}) - ID: {clean_id}")
                time.sleep(1)
                continue
        except Exception as e:
            print(f"⚠️ 본문 조회 실패 ({news_id}): {e}")
            break

    # 모든 시도 실패 시
    print(f"💀 최종 실패: 본문을 가져올 수 없습니다. (ID: {clean_id})")
    return None

# ==========================================
# PDF 파일 로더 함수
# ==========================================
def monitor_pdf_directory(base_directory_path):
    """
    시스템 날짜를 자동으로 파악하여 해당 날짜 폴더를 감시합니다.
    폴더가 없으면 생성하며, 새로운 PDF를 처리 후 OLD 폴더로 이동합니다.
    """

    # 루프 시작 전 현재 날짜 확인
    today_str = datetime.now().strftime("%Y%m%d")
    current_watch_path = Path(base_directory_path) / today_str

    print(f"👀 [PDF Monitor] 오늘 날짜({today_str}) 폴더 감시 시작")
    print(f"📍 [Path] {current_watch_path}")

    # 3. 무한 루프로 폴더 감시
    while True:
        try:
            # 1. 자정이 지나 날짜가 바뀌었는지 체크
            new_today_str = datetime.now().strftime("%Y%m%d")
            if new_today_str != today_str:
                today_str = new_today_str
                current_watch_path = Path(base_directory_path) / today_str
                print(f"📅 [PDF Monitor] 날짜 변경 감지: {today_str} 폴더로 전환")

            # 2. 감시 경로 존재 확인 및 생성
            if not current_watch_path.exists():
                current_watch_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 [PDF Monitor] 새 날짜 폴더 생성됨: {current_watch_path}")

            # 3. OLD 폴더 생성
            old_dir_path = current_watch_path / "OLD"
            old_dir_path.mkdir(parents=True, exist_ok=True)


            # 현재 폴더에 있는 PDF 파일만 검색 (iterdir() 사용으로 한글/공백 대응 강화)
            pdf_files = list(current_watch_path.glob("*.pdf"))

            if pdf_files:
                print(f"🔎 [PDF Monitor] {len(pdf_files)}개의 신규 PDF 발견")

            for file_path in pdf_files:
                filename = file_path.name
                title = file_path.stem  # 확장자 제외 파일명

                # 개별 파일 처리 try-except
                try:
                    # [추가] 안전한 경로 체크 함수 활용
                    # 폴더명에서 날짜 추출 (기존 로직 유지)

                    # date_folder = base_path.name
                    date_folder = current_watch_path.name
                    default_date = date_folder if len(date_folder) == 8 and date_folder.isdigit() else datetime.now().strftime("%Y%m%d")

                    # 파일 읽기 시도
                    full_text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_text += page_text

                    if not full_text.strip():
                        print(f"⚠️ [PDF Loader] 텍스트 없음 (Skip & Move): {filename}")
                    else:
                        # 큐에 넣을 데이터 구조 생성
                        news_data = {
                            "date": default_date,
                            "time": datetime.now().strftime("%H%M%S"),
                            "realkey": f"PDF_{filename}_{int(time.time())}",
                            "id": filename,
                            "title": title,
                            "body": None,
                            "full_text": full_text,
                            "source": "PDF"
                        }
                        news_queue.put(news_data)
                        print(f"📥 [PDF Loader] 큐에 추가됨: {title}")

                    # 4. 처리 완료 후 파일 이동 (shutil 사용 시 경로를 문자열로 변환)
                    destination = old_dir_path / filename

                    # 중복 파일명 처리 (파일명_timestamp.pdf)
                    if destination.exists():
                        new_name = f"{destination.stem}_{int(time.time())}{destination.suffix}"
                        destination = old_dir_path / new_name

                    # [중요] 이동 시 shutil.move(Path객체, Path객체)는 파이썬 3.6+ 에서 지원
                    shutil.move(str(file_path), str(destination))
                    print(f"🚚 [PDF Monitor] 이동 완료 -> OLD/{destination.name}")

                except PermissionError:
                    print(f"🔒 [PDF Monitor] 파일 사용 중 (Skip): {filename}")
                except Exception as e:
                    print(f"❌ [PDF Monitor] 파일 처리 실패 ({filename}): {e}")

            # 5. 폴더 스캔 주기 (3초 대기)
            time.sleep(3)

        except Exception as e:
            print(f"⚠️ [PDF Monitor] 루프 에러: {e}")
            time.sleep(5)

    print("✅ [PDF Loader] 모든 PDF 파일 로딩 완료")




# ==========================================
# 7. 워커 스레드
# ==========================================

def worker():
    """대기열에서 뉴스를 꺼내 처리하는 소비자 함수"""
    print("🚀 뉴스 처리 워커(Worker) 시작됨...")
    while True:
        news_item = news_queue.get()
        try:
            if news_item is None:
                print("\n🛑 워커 종료 신호 수신")
                return

            # [부하 제어] 큐가 많이 차있으면 키워드 없는 뉴스는 스킵
            is_busy = news_queue.qsize() > (MAX_QUEUE_SIZE * 0.7)

            raw_title = news_item.get('title', 'No Title')
            title = clean_financial_text(raw_title)

            # PDF인지 WebSocket인지 확인
            source = news_item.get('source', 'API')
            realkey = news_item.get('realkey')
            news_id = news_item.get('id')

            # debug (사용자 요청: 기존 print 유지)
            print(f"\n날짜: {news_item.get('date')}")
            print(f"시간: {news_item.get('time')}")
            print(f"키값: {news_item.get('realkey')}")
            print(f"제목: {news_item.get('title')}")

            raw_date = news_item.get('date')
            raw_time = news_item.get('time')
            date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if raw_date and len(raw_date) == 8 else raw_date
            time_str = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:]}" if raw_time and len(raw_time) == 6 else raw_time

            # 키워드 필터링 (부하가 심할 때만)
            if is_busy and not quick_keyword_classify(title):
                print(f"⏩ [Skip] 시스템 부하로 건너뜀: {title}")
                # news_queue.task_done()
                continue

            # API 요청 간격 조절
            print(f"\n🔄 처리 시작: {title}")
            if source == "PDF":
                # PDF는 이미 텍스트를 가지고 있음
                raw_full_text = news_item.get('full_text', "")
                # PDF 텍스트도 기본 정제 수행
                # merge_news_bodies는 List[Dict]용이므로 호출하면 안됨
                # joined_full_text = merge_news_bodies(raw_full_text)
                cleaned_full_text = clean_base_text(raw_full_text)
                raw_body = refine_financial_structure(cleaned_full_text)

            else:
                time.sleep(0.5)  # 0.5초 정도 숨 고르기 후 요청

                # 1. 본문 가져오기
                raw_body = fetch_news_body(realkey)

            if raw_body:
                print("\n뉴스 본문:")
                print(raw_body + "\n")

                # 2. 요약 + 분류 (OpenAI API 사용)
                summary_body, category = summarize_and_classify(raw_body, title)

                if not summary_body or not summary_body.strip():
                    print("⚠️ [Fallback Summary] LLM 요약이 비어있어 원문 일부로 대체 저장합니다.")
                    summary_body = (raw_body[:2000] + " ...") if raw_body else "요약 실패/원문 없음"

                db_data = (
                    date,
                    time_str,
                    news_id,
                    realkey,
                    title,
                    len(raw_body),
                    category,
                    summary_body
                )

                # 3. DB 저장
                if True: # "기타" not in category # and "주도 섹터" not in category
                    insert_to_db(db_data)
                    print(f"\n✅ DB 저장 완료: {title} (카테고리: {category})")
                else:
                    print(f"\n🚫 '{category}' 카테고리로 분류되어 저장하지 않습니다: {title}\n")
            else:
                print("\n⚠️ 본문 없음, 건너뜀.")

        except Exception as e:
            print(f"\n❌ 워커 처리 중 에러: {e}")
        finally:
            news_queue.task_done()


# ==========================================
# 8. WebSocket 이벤트 핸들러
# ==========================================

def on_message(ws, message):
    try:
        response = json.loads(message)

        # body 키가 있고, 그 값이 None이 아닐 때만 처리
        if "body" in response and response["body"] is not None:
            news_data = response["body"]

            # [추가] 필수 데이터(제목 등)가 있는지 한 번 더 확인 (안전장치)
            if not isinstance(news_data, dict): return

            try:
                # 큐가 꽉 차면 예외 발생 (데이터 버림)
                news_queue.put(news_data, block=False)

                # news_queue.put(news_data)
                print(f"📩 [수신] {news_data.get('title', '제목없음')} -> 대기열 추가됨")
            except queue.Full:
                print(f"🔥 [Drop] 큐 가득 참! 뉴스 버림: {news_data.get('title')}")
        else:
            # body가 없거나 None인 경우 (Heartbeat나 시스템 메시지일 수 있음)
            pass


    except Exception as e:
        print(f"메시지 파싱 에러: {e}")


# [수정 전]
# def on_open(ws):
#     print("🌐 WebSocket 연결 및 구독 요청")
#     sub_msg = {
#         "header": {"token": ACCESS_TOKEN, "tr_type": "3"},  <-- 여기서 ACCESS_TOKEN은 None입니다.
#         "body": {"tr_cd": "NWS", "tr_key": "NWS001"}
#     }
#     ws.send(json.dumps(sub_msg))

# [수정 후]
def on_open(ws):
    print("🌐 WebSocket 연결 및 구독 요청")

    # 1. 유효한 토큰을 가져옵니다. (만료되었으면 자동 갱신됨)
    token = issue_ls_access_token(force=False)

    if not token:
        print("❌ [Error] 웹소켓 구독 실패: 토큰 발급 불가")
        return

    sub_msg = {
        "header": {"token": token, "tr_type": "3"},  # 2. 발급받은 token 변수 사용
        "body": {"tr_cd": "NWS", "tr_key": "NWS001"}
    }
    ws.send(json.dumps(sub_msg))


# ==========================================
# 9. 메인 실행부
# ==========================================

if __name__ == "__main__":
    # 워커 스레드 시작
    # 워커 수 설정 (로컬 LLM 사용 시 1~2개가 적절)
    num_workers = 1
    print(f"🧵 워커 스레드 수: {num_workers}")
    # num_workers = min(3, max(1, n_cpu_cores - 2))

    # print(f"🧵 워커 스레드 수: {num_workers}")

    worker_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        worker_threads.append(t)

    # PDF 파일 로더 실행 (별도 스레드 혹은 시작 전 실행)
    # WebSocket과 동시에 돌리려면 스레드로,
    pdf_thread = threading.Thread(target=monitor_pdf_directory, args=(PDF_BASE_PATH,), daemon=True)
    pdf_thread.start()

    # 웹소켓 실행 (라이브 뉴스도 계속 받고 싶다면 유지)
    # 웹소켓 디버그 로그 끄기
    websocket.enableTrace(False)

    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_open=on_open,
        on_close=lambda ws, status_cd, msg: print("WebSocket 연결 종료:", status_cd, msg),
        on_error=lambda ws, error: print("WebSocket 에러:", error)
    )

    try:
        print("\n📡 WebSocket 클라이언트 실행 중...")
        ws_app.run_forever()

    except KeyboardInterrupt:
        print("프로그램 종료 중...")
        for _ in worker_threads:
            news_queue.put(None)
        for t in worker_threads:
            t.join(timeout=2.0) # 스레드가 종료될 때까지 최대 2초만 기다림.
