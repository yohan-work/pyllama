.PHONY: setup run clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	@echo "✅ 가상환경 생성 및 패키지 설치 완료"
	@echo "💡 다음 명령으로 실행하세요: make run"

run:
	@if [ ! -d ".venv" ]; then echo "❌ 가상환경이 없습니다. 먼저 'make setup'을 실행하세요."; exit 1; fi
	.venv/bin/python main.py

clean:
	rm -rf .venv
	@echo "✅ 가상환경 삭제 완료"
