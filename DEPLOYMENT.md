# 🚀 Streamlit Cloud 배포 완벽 가이드

## 📋 배포 전 체크리스트

### ✅ 필수 준비사항
- [ ] GitHub 계정 생성
- [ ] OpenAI API 키 발급
- [ ] 모든 파일이 프로젝트 폴더에 준비됨
- [ ] 로컬 테스트 완료

## 🔧 단계별 배포 과정

### 1단계: GitHub 저장소 생성

1. **GitHub 로그인**
   - https://github.com 접속
   - 계정 로그인

2. **새 저장소 생성**
   - "New repository" 클릭
   - Repository name: `rag-chatbot` (또는 원하는 이름)
   - Description: `Langchain RAG 챗봇 프로젝트`
   - Public 선택 (Streamlit Cloud 무료 배포를 위해)
   - "Create repository" 클릭

3. **파일 업로드**
   ```bash
   # Git 초기화 (선택사항)
   git init
   git add .
   git commit -m "Initial commit: RAG chatbot project"
   git branch -M main
   git remote add origin https://github.com/[사용자명]/rag-chatbot.git
   git push -u origin main
   ```

### 2단계: Streamlit Cloud 연결

1. **Streamlit Cloud 접속**
   - https://share.streamlit.io 접속
   - GitHub 계정으로 로그인

2. **새 앱 생성**
   - "New app" 클릭
   - Repository: `[사용자명]/rag-chatbot` 선택
   - Branch: `main` 선택
   - Main file path: `app.py` 입력
   - App URL: 원하는 URL 입력 (예: `my-rag-chatbot`)

### 3단계: 환경 변수 설정

1. **Secrets 설정**
   - 앱 설정 페이지에서 "Secrets" 섹션 찾기
   - 다음 내용 입력:

   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

2. **고급 설정 (선택사항)**
   ```toml
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_API_KEY = "your-langchain-api-key"
   LANGCHAIN_PROJECT = "rag-chatbot"
   ```

### 4단계: 배포 실행

1. **배포 시작**
   - "Deploy!" 버튼 클릭
   - 배포 과정 모니터링 (약 2-3분 소요)

2. **배포 완료 확인**
   - 성공 시 제공되는 URL로 접속
   - 앱이 정상적으로 로드되는지 확인

## 🔍 배포 후 테스트

### 기본 기능 테스트
1. **문서 업로드 테스트**
   - 샘플 PDF 파일 업로드
   - 로딩 과정 확인

2. **질의응답 테스트**
   - 간단한 질문 입력
   - 답변 품질 확인
   - 참조 문서 표시 확인

3. **UI 테스트**
   - 반응형 디자인 확인
   - 모든 버튼 동작 확인
   - 에러 메시지 표시 확인

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. 배포 실패
**원인**: 패키지 의존성 문제
**해결방법**:
```bash
# requirements.txt 확인
pip install -r requirements.txt
```

#### 2. API 키 오류
**원인**: OpenAI API 키 미설정 또는 잘못된 키
**해결방법**:
- OpenAI 웹사이트에서 API 키 재발급
- Streamlit Cloud에서 환경 변수 재설정

#### 3. 메모리 부족
**원인**: 큰 문서 파일 처리
**해결방법**:
- 문서 크기 제한 (10MB 이하)
- 청크 크기 조정

#### 4. 로딩 시간 지연
**원인**: 첫 로딩 시 모델 다운로드
**해결방법**:
- 첫 로딩은 시간이 걸릴 수 있음 (정상)
- 이후 로딩은 빠름

### 성능 최적화 팁

1. **문서 크기 최적화**
   - PDF 파일: 10MB 이하 권장
   - 텍스트 파일: 5MB 이하 권장

2. **청크 크기 조정**
   ```python
   # app.py에서 조정 가능
   chunk_size=1000  # 기본값
   chunk_overlap=200  # 기본값
   ```

3. **임베딩 모델 변경**
   ```python
   # 더 빠른 모델 사용
   model_name="sentence-transformers/all-MiniLM-L6-v2"
   ```

## 📊 배포 상태 모니터링

### Streamlit Cloud 대시보드
- **앱 상태**: 실행 중/중지 상태 확인
- **사용량**: CPU, 메모리 사용량 모니터링
- **로그**: 오류 로그 및 디버깅 정보 확인

### 성능 지표
- **응답 시간**: 평균 3-5초
- **동시 사용자**: 최대 100명
- **가용성**: 99.9% 이상

## 🔄 업데이트 및 유지보수

### 코드 업데이트
1. 로컬에서 코드 수정
2. GitHub에 푸시
3. Streamlit Cloud에서 자동 재배포

### 환경 변수 변경
1. Streamlit Cloud 설정 페이지 접속
2. Secrets 섹션에서 수정
3. 앱 재시작

## 📱 모바일 최적화

### 반응형 디자인 확인
- 다양한 화면 크기에서 테스트
- 터치 인터페이스 동작 확인
- 로딩 시간 모바일에서 확인

## 🎯 최종 체크리스트

배포 완료 후 다음 사항들을 확인하세요:

- [ ] 앱이 정상적으로 로드됨
- [ ] 문서 업로드 기능 동작
- [ ] 질의응답 기능 동작
- [ ] 참조 문서 표시 기능 동작
- [ ] 에러 처리 기능 동작
- [ ] 모바일에서 정상 동작
- [ ] 로딩 시간 적절함
- [ ] UI 디자인 만족스러움

## 🎉 배포 완료!

축하합니다! RAG 챗봇이 성공적으로 배포되었습니다.

**배포된 URL**: `https://[앱이름]-[사용자명].streamlit.app`

이제 누구나 인터넷을 통해 여러분의 RAG 챗봇을 사용할 수 있습니다!

---

**참고사항**:
- 무료 계정의 경우 월 1,000시간 제한
- 유료 계정 업그레이드 시 더 많은 기능 이용 가능
- 정기적인 모니터링으로 안정성 유지 권장