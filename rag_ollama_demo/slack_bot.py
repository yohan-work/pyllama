#!/usr/bin/env python3
"""
슬랙 RAG 봇 - Ollama + Chroma 기반
"""

import os
import re
import time
from typing import List, Dict, Any
import asyncio
from dotenv import load_dotenv

# 슬랙 SDK 
from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

# 기존 RAG 시스템 임포트
from main import build_retriever, build_chain, is_relevant_to_docs
from langchain_community.llms import Ollama

# 환경변수 로드
load_dotenv()

# 설정
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.8"))

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise ValueError("SLACK_BOT_TOKEN과 SLACK_APP_TOKEN을 .env 파일에 설정해주세요!")

# 슬랙 클라이언트 초기화
web_client = WebClient(token=SLACK_BOT_TOKEN)
socket_client = SocketModeClient(app_token=SLACK_APP_TOKEN, web_client=web_client)

# RAG 시스템 전역 변수
retriever = None
qa_chain = None
llm = None

def initialize_rag_system():
    """RAG 시스템 초기화"""
    global retriever, qa_chain, llm
    
    print("🚀 RAG 시스템 초기화 중...")
    try:
        # 기존 main.py의 로직 가져오기
        from main import load_raw_texts, DOC_DIR
        
        # 1. 문서 로드
        print("📚 문서 로딩 중...")
        texts = load_raw_texts(DOC_DIR)
        if not texts:
            print("⚠️  경고: docs/ 폴더에 문서가 없습니다.")
            
        # 2. 리트리버 구축
        retriever = build_retriever(texts)
        print("✅ 문서 인덱싱 완료")
        
        # 3. QA 체인 구축 
        qa_chain, llm = build_chain(retriever)
        print("✅ QA 체인 구축 완료")
        print("✅ LLM 연결 완료")
        
        print("🎉 RAG 시스템 준비 완료!")
        return True
        
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        return False

def clean_text(text: str) -> str:
    """텍스트 정리 (슬랙 마크다운 제거 등)"""
    # 슬랙 사용자 멘션 제거 (<@U12345> 형태)
    text = re.sub(r'<@U\w+>', '', text)
    # 슬랙 채널 멘션 제거 (<#C12345|channel-name> 형태)
    text = re.sub(r'<#C\w+\|[\w-]+>', '', text)
    # 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_user_question(question: str) -> Dict[str, Any]:
    """사용자 질문 처리"""
    start_time = time.time()
    
    try:
        # 텍스트 정리
        clean_question = clean_text(question)
        
        if not clean_question:
            return {
                "answer": "질문을 명확히 입력해주세요! 🤔",
                "is_rag": False,
                "sources": [],
                "response_time": time.time() - start_time
            }
        
        # 관련성 판단 시작 시간
        relevance_start = time.time()
        is_relevant, relevant_docs = is_relevant_to_docs(clean_question, retriever, SIMILARITY_THRESHOLD)
        relevance_time = time.time() - relevance_start
        print(f"[TIMING] 관련성 판단: {relevance_time:.2f}초")
        
        if is_relevant:
            # 📚 RAG 답변
            print(f"[INFO] RAG 모드로 답변: {clean_question}")
            rag_start = time.time()
            result = qa_chain.invoke({"query": clean_question})
            rag_time = time.time() - rag_start
            print(f"[TIMING] RAG 생성: {rag_time:.2f}초")
            
            # 소스 문서 정보 추출
            sources = []
            if hasattr(result, 'source_documents'):
                for i, doc in enumerate(result.source_documents[:3], 1):
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    sources.append(f"{i}. {content_preview}...")
            
            total_time = time.time() - start_time
            print(f"[TIMING] 전체 RAG 응답: {total_time:.2f}초")
            
            return {
                "answer": result["result"],
                "is_rag": True,
                "sources": sources,
                "response_time": total_time
            }
        else:
            # 🤖 순수 LLM 답변
            print(f"[INFO] LLM 모드로 답변: {clean_question}")
            llm_start = time.time()
            korean_prompt = f"다음 질문에 한국어로 답변해주세요: {clean_question}"
            answer = llm.invoke(korean_prompt)
            llm_time = time.time() - llm_start
            print(f"[TIMING] LLM 생성: {llm_time:.2f}초")
            
            total_time = time.time() - start_time
            print(f"[TIMING] 전체 LLM 응답: {total_time:.2f}초")
            
            return {
                "answer": answer,
                "is_rag": False,
                "sources": [],
                "response_time": total_time
            }
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"[ERROR] 질문 처리 중 오류 ({total_time:.2f}초): {e}")
        return {
            "answer": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)} 😅",
            "is_rag": False,
            "sources": [],
            "response_time": total_time
        }

def format_slack_response(result: Dict[str, Any]) -> str:
    """슬랙 응답 포맷팅"""
    answer = result["answer"]
    is_rag = result["is_rag"]
    sources = result["sources"]
    response_time = result.get("response_time", 0)
    
    # 기본 답변
    response = f"{answer}\n\n"
    
    # 답변 타입 및 응답 시간 표시
    if is_rag:
        response += f"📚 *학습된 문서 기반 답변* (⏱️ {response_time:.2f}초)\n"
        if sources:
            response += "*근거 출처:*\n"
            for source in sources:
                response += f"• {source}\n"
    else:
        response += f"🤖 *일반 지식 기반 답변* (⏱️ {response_time:.2f}초)"
    
    return response

def process_slack_message(client: SocketModeClient, req: SocketModeRequest):
    """슬랙 메시지 처리"""
    
    print(f"[DEBUG] 이벤트 수신: {req.type}")
    
    # 응답 확인
    response = SocketModeResponse(envelope_id=req.envelope_id)
    client.send_socket_mode_response(response)
    
    # 이벤트 데이터 추출
    event = req.payload.get("event", {})
    event_type = event.get("type")
    
    print(f"[DEBUG] 이벤트 타입: {event_type}")
    print(f"[DEBUG] 이벤트 데이터: {event}")
    
    # 봇 자신의 메시지는 무시
    if event.get("bot_id"):
        print(f"[DEBUG] 봇 메시지 무시")
        return
        
    # 메시지 이벤트 처리
    if event_type in ["message", "app_mention"]:
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "")
        
        if not text:
            return
            
        print(f"[INFO] 메시지 수신: {text}")
        
        # 로딩 메시지 표시
        try:
            loading_response = web_client.chat_postMessage(
                channel=channel_id,
                text="🤔 생각 중... 잠시만 기다려주세요!"
            )
            loading_ts = loading_response["ts"]
        except Exception as e:
            print(f"[WARN] 로딩 메시지 표시 실패: {e}")
            loading_ts = None
        
        # 질문 처리
        result = process_user_question(text)
        formatted_response = format_slack_response(result)
        
        # 최종 답변으로 업데이트
        try:
            if loading_ts:
                # 로딩 메시지를 실제 답변으로 업데이트
                web_client.chat_update(
                    channel=channel_id,
                    ts=loading_ts,
                    text=formatted_response
                )
            else:
                # 새 메시지로 답변
                web_client.chat_postMessage(
                    channel=channel_id,
                    text=formatted_response
                )
        except Exception as e:
            print(f"[ERROR] 답변 전송 실패: {e}")
            # 백업 답변
            web_client.chat_postMessage(
                channel=channel_id,
                text="죄송합니다. 답변 전송 중 오류가 발생했습니다. 😅"
            )

def main():
    """메인 실행 함수"""
    print("🔥 슬랙 RAG 봇 시작!")
    
    # RAG 시스템 초기화
    if not initialize_rag_system():
        print("❌ RAG 시스템 초기화 실패. 종료합니다.")
        return
    
    # 슬랙 이벤트 핸들러 등록
    socket_client.socket_mode_request_listeners.append(process_slack_message)
    
    print("🎯 슬랙 봇이 준비되었습니다!")
    print("슬랙에서 봇을 멘션(@RAG Assistant Bot)하거나 DM을 보내보세요!")
    
    # 봇 시작
    try:
        socket_client.connect()
        print("🔗 슬랙에 연결되었습니다!")
        
        # 무한 대기
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 봇을 종료합니다...")
    except Exception as e:
        print(f"❌ 봇 실행 중 오류: {e}")
    finally:
        socket_client.disconnect()

if __name__ == "__main__":
    main()
