#!/usr/bin/env python3
"""
RAG 챗봇 테스트 스크립트
로컬에서 앱을 실행하고 기본 기능을 테스트합니다.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_requirements():
    """필요한 패키지가 설치되어 있는지 확인"""
    print("📦 필요한 패키지 확인 중...")
    
    try:
        import streamlit
        import langchain
        import chromadb
        import openai
        print("✅ 모든 필수 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False

def check_files():
    """필요한 파일들이 존재하는지 확인"""
    print("📁 파일 구조 확인 중...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".streamlit/config.toml",
        ".streamlit/secrets.toml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 다음 파일들이 없습니다: {missing_files}")
        return False
    else:
        print("✅ 모든 필요한 파일이 존재합니다.")
        return True

def test_app_startup():
    """앱 시작 테스트"""
    print("🚀 앱 시작 테스트 중...")
    
    try:
        # Streamlit 앱을 백그라운드에서 시작
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.headless", "true", "--server.port", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 앱이 시작될 때까지 대기
        time.sleep(10)
        
        # 앱이 실행 중인지 확인
        try:
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ 앱이 성공적으로 시작되었습니다.")
                process.terminate()
                return True
            else:
                print(f"❌ 앱 시작 실패: HTTP {response.status_code}")
                process.terminate()
                return False
        except requests.exceptions.RequestException:
            print("❌ 앱에 연결할 수 없습니다.")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ 앱 시작 중 오류 발생: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🤖 RAG 챗봇 테스트 시작")
    print("=" * 50)
    
    # 1. 패키지 확인
    if not check_requirements():
        return False
    
    # 2. 파일 확인
    if not check_files():
        return False
    
    # 3. 앱 시작 테스트
    if not test_app_startup():
        return False
    
    print("=" * 50)
    print("🎉 모든 테스트가 통과했습니다!")
    print("\n📋 다음 단계:")
    print("1. OpenAI API 키를 .streamlit/secrets.toml에 설정")
    print("2. streamlit run app.py 명령어로 앱 실행")
    print("3. 브라우저에서 http://localhost:8501 접속")
    print("4. 샘플 문서(sample_document.txt) 업로드하여 테스트")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
