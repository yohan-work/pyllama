# RAG + Ollama + Chroma (Python) Minimal Demo

## 0) 준비

- 로컬에 [Ollama](https://ollama.com) 설치 및 실행
- 한 번만 모델 풀: `ollama pull llama3`

## 1) 실행

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

## 2) 문서 추가

docs/ 폴더에 .txt / .md / .pdf 파일을 자유롭게 넣으세요.

다시 python main.py 실행 시 자동 인덱싱합니다.

## 3) 환경 변수 (선택)

```bash
# 모델/청크/탐색 파라미터 조정 가능
export OLLAMA_MODEL=llama3
export EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export CHUNK_SIZE=200                    # 청크 크기 (작을수록 정밀)
export CHUNK_OVERLAP=50                  # 청크 겹침
export TOP_K=3                          # 검색할 문서 수
export TEMPERATURE=0.7                  # LLM 창의성 (0.0-1.0)
export SIMILARITY_THRESHOLD=0.3         # 관련성 임계값 (낮을수록 엄격)
```

### 🎯 자동 관련성 판단

- **키워드 하드코딩 없음**: 벡터 유사도 점수로 자동 판단
- **SIMILARITY_THRESHOLD**: 0.0-1.0 (낮을수록 엄격, 높을수록 관대)
- **자동 모드 전환**: 관련성에 따라 RAG ↔ LLM 자동 선택

## 4) 사용법

프로그램 실행 후 `질문>` 프롬프트에 한국어로 질문을 입력하세요.
답변 하단에 검색 근거 스니펫이 함께 표시됩니다.
