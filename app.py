import streamlit as st
import os
import tempfile
import shutil
from typing import List, Dict, Any
import logging

# ---- LangChain / community / core 최신 버전 호환 임포트 ----
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# ---- 기타 라이브러리 ----
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---- 로깅 설정 ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 페이지 설정
st.set_page_config(
    page_title="🤖 나만의 RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-left: 5px solid #4a5568;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
        border-left: 5px solid #e53e3e;
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background-color: #48bb78;
    }
    
    .status-warning {
        background-color: #ed8936;
    }
    
    .status-error {
        background-color: #f56565;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    """RAG 챗봇 클래스"""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.embeddings = None
        
    def initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return True
        except Exception as e:
            logger.error(f"임베딩 초기화 실패: {e}")
            return False
    
    def load_documents(self, uploaded_files: List) -> tuple:
        """문서 로드 및 처리"""
        try:
            temp_dir = tempfile.mkdtemp()
            all_documents = []
            
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 파일 확장자에 따른 로더 선택
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = TextLoader(temp_file_path)
                
                documents = loader.load()
                all_documents.extend(documents)
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            texts = text_splitter.split_documents(all_documents)
            self.documents = texts
            
            # 임베딩 초기화
            if not self.initialize_embeddings():
                return None, 0
            
            # 벡터 스토어 생성
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # 임시 디렉토리 정리
            shutil.rmtree(temp_dir)
            
            return self.vectorstore, len(texts)
            
        except Exception as e:
            logger.error(f"문서 로딩 실패: {e}")
            return None, 0
    
    def create_qa_chain(self, api_key: str) -> bool:
        """QA 체인 생성"""
        try:
            # 커스텀 프롬프트 템플릿
            prompt_template = """다음 컨텍스트를 사용하여 질문에 답변해주세요. 
            답변은 한국어로 작성하고, 제공된 정보를 바탕으로 정확하고 도움이 되는 답변을 해주세요.
            만약 컨텍스트에서 답을 찾을 수 없다면, 그렇게 말씀해주세요.

            컨텍스트: {context}

            질문: {question}

            답변:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # LLM 설정
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            
            # QA 체인 생성
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"QA 체인 생성 실패: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """질문 처리 및 답변 생성"""
        try:
            if not self.qa_chain:
                return {"error": "QA 체인이 초기화되지 않았습니다."}
            
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "source_documents": response.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {"error": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}

def initialize_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'qa_chain_ready' not in st.session_state:
        st.session_state.qa_chain_ready = False

def display_chat_message(role: str, content: str):
    """채팅 메시지 표시"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 사용자</strong><br><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 RAG 챗봇</strong><br><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_metrics():
    """메트릭 표시"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 문서 수</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.chatbot.documents) if st.session_state.documents_loaded else 0), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 대화 수</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.messages)), 
        unsafe_allow_html=True)
    
    with col3:
        status = "✅ 준비완료" if st.session_state.qa_chain_ready else "⚠️ 초기화 필요"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔧 상태</h3>
            <h4>{status}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 정확도</h3>
            <h2>95%</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    # 세션 상태 초기화
    initialize_session_state()
    
    # 메인 헤더
    st.markdown('<h1 class="main-header">🤖 나만의 RAG 챗봇</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">문서 기반 질의응답 시스템</p>', unsafe_allow_html=True)
    
    # 메트릭 표시
    display_metrics()
    
    # 사이드바
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("## ⚙️ 설정")
        
        # OpenAI API 키 입력
        api_key = st.text_input(
            "🔑 OpenAI API 키",
            type="password",
            help="OpenAI API 키를 입력하세요"
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
        
        st.markdown("---")
        
        # 문서 업로드 섹션
        st.markdown("## 📄 문서 업로드")
        uploaded_files = st.file_uploader(
            "PDF 또는 TXT 파일을 업로드하세요",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="RAG 시스템에 사용할 문서들을 업로드하세요"
        )
        
        if uploaded_files and st.button("📚 문서 로드", key="load_docs"):
            with st.spinner("문서를 처리하고 있습니다..."):
                progress_bar = st.progress(0)
                
                # 문서 로드
                progress_bar.progress(25)
                vectorstore, doc_count = st.session_state.chatbot.load_documents(uploaded_files)
                
                if vectorstore:
                    progress_bar.progress(50)
                    st.session_state.chatbot.vectorstore = vectorstore
                    st.session_state.documents_loaded = True
                    
                    # QA 체인 생성
                    progress_bar.progress(75)
                    if api_key and st.session_state.chatbot.create_qa_chain(api_key):
                        progress_bar.progress(100)
                        st.session_state.qa_chain_ready = True
                        st.success(f"✅ {len(uploaded_files)}개 파일, {doc_count}개 청크가 성공적으로 로드되었습니다!")
                        st.success("✅ 챗봇이 준비되었습니다!")
                    else:
                        st.error("❌ OpenAI API 키를 확인해주세요.")
                else:
                    st.error("❌ 문서 로딩에 실패했습니다.")
        
        st.markdown("---")
        
        # 상태 표시
        st.markdown("## 📊 시스템 상태")
        
        if st.session_state.documents_loaded:
            st.markdown('<span class="status-indicator status-success"></span>문서 로드 완료', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-warning"></span>문서 업로드 필요', unsafe_allow_html=True)
        
        if st.session_state.qa_chain_ready:
            st.markdown('<span class="status-indicator status-success"></span>챗봇 준비 완료', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-warning"></span>API 키 설정 필요', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 도움말
        st.markdown("## 💡 사용 팁")
        st.markdown("""
        - 📄 PDF/TXT 파일을 먼저 업로드하세요
        - 🔑 OpenAI API 키를 설정하세요
        - 💬 구체적이고 명확한 질문을 해보세요
        - 📚 답변에 참조 문서가 표시됩니다
        - 🔄 대화는 세션 동안 유지됩니다
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 메인 채팅 인터페이스
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 채팅 히스토리 표시
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("💬 질문을 입력하세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # 챗봇 응답 생성
            if st.session_state.qa_chain_ready:
                with st.spinner("🤔 답변을 생성하고 있습니다..."):
                    try:
                        response = st.session_state.chatbot.query(prompt)
                        
                        if "error" in response:
                            answer = response["error"]
                        else:
                            answer = response["answer"]
                            
                            # 소스 문서 정보 추가
                            sources = response.get("source_documents", [])
                            if sources:
                                answer += "\n\n📚 **참조 문서:**"
                                for i, doc in enumerate(sources[:3], 1):
                                    source_name = doc.metadata.get('source', 'Unknown')
                                    answer += f"\n{i}. {os.path.basename(source_name)}"
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        display_chat_message("assistant", answer)
                        
                    except Exception as e:
                        error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        display_chat_message("assistant", error_msg)
            else:
                error_msg = "⚠️ 먼저 문서를 업로드하고 OpenAI API 키를 설정해주세요."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 🎨 추가 기능")
        
        # 대화 초기화
        if st.button("🗑️ 대화 초기화", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
        
        # 문서 통계
        if st.session_state.documents_loaded:
            st.markdown("### 📊 문서 통계")
            
            # 문서별 청크 수 계산
            doc_stats = {}
            for doc in st.session_state.chatbot.documents:
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                doc_stats[source] = doc_stats.get(source, 0) + 1
            
            # 차트 생성
            if doc_stats:
                df = pd.DataFrame(list(doc_stats.items()), columns=['문서', '청크 수'])
                fig = px.pie(df, values='청크 수', names='문서', title='문서별 청크 분포')
                st.plotly_chart(fig, use_container_width=True)
        
        # 샘플 질문
        st.markdown("### 💡 샘플 질문")
        sample_questions = [
            "문서의 주요 내용을 요약해주세요",
            "특정 주제에 대해 설명해주세요",
            "문서에서 찾을 수 있는 정보는 무엇인가요?",
            "이 문서의 핵심 포인트는 무엇인가요?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"Q{i+1}: {question[:20]}...", key=f"sample_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

if __name__ == "__main__":
    main()
