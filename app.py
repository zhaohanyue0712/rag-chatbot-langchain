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
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# ---- 기타 라이브러리 ----
import pandas as pd
import plotly.express as px

# ---- 로깅 설정 ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(
    page_title="🤖 나만의 RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 커스텀 CSS
# ---------------------------
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
    .chat-container {
        background: #f5f7fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }
    .user-message {
        background: #667eea;
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background: #f093fb;
        color: white;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# RAG 챗봇 클래스
# ---------------------------
class RAGChatbot:
    """RAG 챗봇 클래스"""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.embeddings = None
        
    def initialize_embeddings(self):
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
        """문서 로드 및 처리 (Streamlit Cloud 환경 호환 버전)"""
        try: 
            import pypdf  # PDF 파싱 백업
            temp_dir = tempfile.mkdtemp()
            all_documents = []

           # 1️⃣ 업로드된 파일을 임시 폴더에 저장
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                  f.write(uploaded_file.getbuffer())

            # 2️⃣ PDF / TXT 로더 (pypdf 폴백)
             try:
                  if uploaded_file.name.lower().endswith(".pdf"):
                     loader = PyPDFLoader(temp_file_path)
                     docs = loader.load()
                  else:
                     loader = TextLoader(temp_file_path, encoding="utf-8")
                     docs = loader.load()
               except Exception:
                  reader = pypdf.PdfReader(temp_file_path)
                  text = "\n".join(page.extract_text() or "" for page in reader.pages)
                  docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

              all_documents.extend(docs)

             # 3️⃣ 텍스트 분할
             if not all_documents:
                 raise RuntimeError("❌ 업로드된 문서에서 텍스트를 추출하지 못했습니다.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(all_documents)
            self.documents = texts

            # 4️⃣ 임베딩 초기화
            if not self.initialize_embeddings():
              raise RuntimeError("❌ 임베딩 초기화 실패")

         # 5️⃣ Chroma 저장소 경로 생성 (/tmp 사용)
          chroma_path = os.path.join(tempfile.gettempdir(), "chroma_db")
          if os.path.exists(chroma_path):
              shutil.rmtree(chroma_path)
          os.makedirs(chroma_path, exist_ok=True)

         # 6️⃣ 벡터 스토어 생성
          self.vectorstore = Chroma.from_documents(
              documents=texts,
              embedding=self.embeddings,
              persist_directory=chroma_path
         )

         # 7️⃣ 임시 폴더 정리
          shutil.rmtree(temp_dir, ignore_errors=True)

          return self.vectorstore, len(texts)

       except Exception as e:
          logger.error(f"문서 로딩 실패: {e}")
          st.error(f"❌ 문서 로딩 중 오류가 발생했습니다: {str(e)}")
          return None, 0

    def create_qa_chain(self, api_key: str) -> bool:
        try:
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
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            return True
        except Exception as e:
            logger.error(f"QA 체인 생성 실패: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        try:
            if not self.qa_chain:
                return {"error": "QA 체인이 초기화되지 않았습니다."}
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "source_documents": response.get("source_documents", [])
            }
        except Exception as e:
            return {"error": f"답변 생성 중 오류: {str(e)}"}

# ---------------------------
# 세션 상태 초기화
# ---------------------------
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'qa_chain_ready' not in st.session_state:
        st.session_state.qa_chain_ready = False

# ---------------------------
# 메시지 표시 함수
# ---------------------------
def display_chat_message(role: str, content: str):
    if role == "user":
        st.markdown(f"<div class='chat-message user-message'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message bot-message'>{content}</div>", unsafe_allow_html=True)

# ---------------------------
# 메인 함수
# ---------------------------
def main():
    initialize_session_state()

    st.markdown('<h1 class="main-header">🤖 나만의 RAG 챗봇</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        api_key = st.text_input("🔑 OpenAI API 키", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
        st.markdown("## 📄 문서 업로드")
        uploaded_files = st.file_uploader("PDF 또는 TXT 파일 업로드", type=['pdf', 'txt'], accept_multiple_files=True)
        if uploaded_files and st.button("📚 문서 로드"):
            with st.spinner("문서를 처리 중..."):
                vectorstore, doc_count = st.session_state.chatbot.load_documents(uploaded_files)
                if vectorstore:
                    st.session_state.chatbot.vectorstore = vectorstore
                    st.session_state.documents_loaded = True
                    if api_key and st.session_state.chatbot.create_qa_chain(api_key):
                        st.session_state.qa_chain_ready = True
                        st.success(f"✅ {doc_count}개 청크 로드 완료!")
                    else:
                        st.error("❌ API 키 오류")
                else:
                    st.error("❌ 문서 로딩 실패")

    # 메인 채팅 UI
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # 입력창 (container 밖)
    prompt = st.chat_input("💬 질문을 입력하세요...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        if st.session_state.qa_chain_ready:
            with st.spinner("🤔 답변 생성 중..."):
                response = st.session_state.chatbot.query(prompt)
                if "error" in response:
                    answer = response["error"]
                else:
                    answer = response["answer"]
                    sources = response.get("source_documents", [])
                    if sources:
                        answer += "\n\n📚 **참조 문서:**"
                        for i, doc in enumerate(sources[:3], 1):
                            source_name = doc.metadata.get('source', 'Unknown')
                            answer += f"\n{i}. {os.path.basename(source_name)}"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                display_chat_message("assistant", answer)
        else:
            msg = "⚠️ 먼저 문서를 업로드하고 API 키를 설정하세요."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            display_chat_message("assistant", msg)

    with col2:
        st.markdown("## ⚙️ 추가 기능")
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    main()
