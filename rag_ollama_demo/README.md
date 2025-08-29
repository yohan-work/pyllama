# RAG + Ollama + Chroma (Python) Minimal Demo

## 0) 준비

- 로컬에 [Ollama](https://ollama.com) 설치 및 실행
- 한 번만 모델 풀: `ollama pull llama3`

## 1) 실행

### 🖥️ 터미널 버전 (기본)

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
python3 main.py
```

### 🤖 슬랙 봇 버전

**1단계: 슬랙 앱 생성**

1. [Slack API](https://api.slack.com/apps) → "Create New App" → "From scratch"
2. 앱 이름: `RAG Assistant Bot`, 워크스페이스 선택

**2단계: 권한 설정**

- **OAuth & Permissions** → **Bot Token Scopes**:
  - `app_mentions:read`, `chat:write`, `channels:read`, `im:read`, `im:write`
- "Install to Workspace" → **Bot User OAuth Token** 복사 (xoxb-로 시작)

**3단계: 이벤트 구독**

- **Event Subscriptions** → "Enable Events" 켜기
- **Subscribe to bot events**: `app_mention`, `message.im`

**4단계: 소켓 모드**

- **Socket Mode** → "Enable Socket Mode" 켜기
- **App-Level Token** 생성 (connections:write 권한) → 토큰 복사 (xapp-로 시작)

**5단계: 환경변수 설정**

```bash
# .env 파일 생성
echo "SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SIMILARITY_THRESHOLD=0.8" > .env
```

**6단계: 슬랙 봇 실행**

```bash
pip install -r requirements.txt
python3 slack_bot.py
```

**사용법:**

- 채널에서 봇 멘션: `@RAG Assistant Bot 안녕하세요!`
- DM으로 직접 질문 가능
- 📚 학습된 문서 관련 질문 → RAG 답변 (근거 출처 표시)
- 🤖 일반 질문 → LLM 답변

## 2) 문서 추가

docs/ 폴더에 다음 형식의 파일들을 자유롭게 넣으세요:

- **텍스트**: `.txt`, `.md`
- **PDF**: `.pdf`
- **Microsoft Office**: `.docx`, `.xlsx`, `.xls`, `.pptx`

다시 python main.py 또는 슬랙 봇 재시작 시 자동 인덱싱합니다.

### 🏢 **회사 NAS 데이터 이관 가능:**

- 회의록 (Word 문서)
- 보고서 (PDF, PowerPoint)
- 데이터 시트 (Excel)
- 매뉴얼 (텍스트, 마크다운)
- 모든 파일이 자동으로 RAG 학습

## 3) 환경 변수 (선택)

```bash
# 모델/청크/탐색 파라미터 조정 가능
export OLLAMA_MODEL=llama3
export EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export CHUNK_SIZE=200                    # 청크 크기 (작을수록 정밀)
export CHUNK_OVERLAP=50                  # 청크 겹침
export TOP_K=3                          # 검색할 문서 수
export TEMPERATURE=0.7                  # LLM 창의성 (0.0-1.0)
export SIMILARITY_THRESHOLD=1.2         # 관련성 임계값 (낮을수록 엄격)
```

### 🎯 자동 관련성 판단

- **키워드 하드코딩 없음**: 벡터 유사도 점수로 자동 판단
- **SIMILARITY_THRESHOLD**: 0.0-2.0 (낮을수록 엄격, 높을수록 관대)
- **자동 모드 전환**: 관련성에 따라 RAG ↔ LLM 자동 선택

## 4) 사용법

프로그램 실행 후 `질문>` 프롬프트에 한국어로 질문을 입력하세요.
답변 하단에 검색 근거 스니펫이 함께 표시됩니다.
