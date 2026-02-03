[기술 보고서] AIO2O - assignment 


(PythonProject) PS V:\PythonProject1\aio2o> python .\main.py                                                                                                                          
INFO:     Started server process [34728]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8088 (Press CTRL+C to quit)



http://127.0.0.1:8088/docs

ㅜ


    # 1. 뉴스 데이터 준비 (예시 데이터 구조 활용)
    # 실제 구현 시 DB fetch_latest_news() 호출
    test_news = [
        {"date": "2026-02-03", "time": "20:41", "title": "워시 연준 의장 후보, 적극적 금리 인하 추진 관측"},
        {"date": "2026-02-03", "time": "16:35", "title": "코스피 사상 최고치 경신, 외국인 대거 순매수"}
    ]


실제 live 코드에서는 Open API 통해서 라이브 뉴스를 웹소켓으로 읽어서 DB에 적재하고 있음. 



    # 기대수익률 벡터 (Call_L, Call_S, Put_L, Put_S, Future)
    if trend == "bullish":
        mu = [0.08, -0.02, -0.06, 0.03, 0.12]
    elif trend == "bearish":
        mu = [-0.07, 0.03, 0.09, -0.02, -0.15]
    else:
        mu = [0.01, 0.01, 0.01, 0.01, 0.0]

실제 라이브 코드에서는 기대수익율 뿐만 아니라, 변동성, correlation 까지 포함해서 포트폴리오 최적화를 수행하고 있음. 
