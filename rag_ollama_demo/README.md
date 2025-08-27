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
export CHUNK_SIZE=600
export CHUNK_OVERLAP=80
export TOP_K=5
export TEMPERATURE=0.2
```

## 4) 사용법

프로그램 실행 후 `질문>` 프롬프트에 한국어로 질문을 입력하세요.
답변 하단에 검색 근거 스니펫이 함께 표시됩니다.
