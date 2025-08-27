#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal RAG with Ollama (local LLM) + Chroma
Run:
  1) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
  2) pip install -r requirements.txt
  3) Make sure Ollama is installed & running: https://ollama.com
     - Pull a model once:  ollama pull llama3
  4) python main.py
"""
import os, glob, sys
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

DOC_DIR = os.environ.get("DOC_DIR", "docs")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3")  # e.g., llama3, qwen2, mistral
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
TOP_K = int(os.environ.get("TOP_K", "5"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_raw_texts(doc_dir: str) -> List[str]:
    texts = []
    for path in sorted(glob.glob(os.path.join(doc_dir, "**", "*.*"), recursive=True)):
        if path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        elif path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                texts.append("\n".join([p.extract_text() or "" for p in reader.pages]))
            except Exception as e:
                print(f"[WARN] PDF 읽기 실패: {path} ({e})")
    return texts

def build_retriever(texts: List[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for t in texts:
        docs.extend(splitter.create_documents([t]))

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(docs, embed)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    return retriever

PROMPT_TMPL = """당신은 도움이 되는 AI 어시스턴트입니다. 
질문에 한국어로 자연스럽게 답변해 주세요.

만약 아래 제공된 컨텍스트가 질문과 관련이 있다면 참고해서 답변하고, 
관련이 없다면 컨텍스트를 무시하고 일반적인 지식으로 답변해 주세요.

컨텍스트:
{context}

질문: {question}

답변:"""

def build_chain(retriever):
    prompt = PromptTemplate(template=PROMPT_TMPL, input_variables=["context","question"])
    llm = Ollama(model=MODEL_NAME, temperature=TEMPERATURE)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

def main():
    if not os.path.isdir(DOC_DIR):
        print(f"[ERR] 문서 폴더가 없습니다: {DOC_DIR}")
        sys.exit(1)

    print(f"[INFO] 문서 인덱싱 중... ({DOC_DIR})")
    texts = load_raw_texts(DOC_DIR)
    if not texts:
        print("[ERR] 인덱싱할 문서가 없습니다. docs 폴더에 txt/md/pdf를 넣으세요.")
        sys.exit(1)

    retriever = build_retriever(texts)
    qa = build_chain(retriever)
    print("[OK] RAG 준비 완료. 아래에 질문을 입력하세요. (종료: 빈 줄 + Enter)")

    while True:
        try:
            q = input("\n질문> ").strip()
            if not q:
                break
            result = qa.invoke({"query": q})
            print("\n--- 답변 ---")
            print(result.get("result","(no result)"))
            srcs = result.get("source_documents", [])
            if srcs:
                print("\n[근거 출처]")
                for i, s in enumerate(srcs, 1):
                    snippet = (s.page_content or "").strip().replace("\n"," ")
                    if len(snippet) > 120:
                        snippet = snippet[:120] + "…"
                    print(f"{i}. {snippet}")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
