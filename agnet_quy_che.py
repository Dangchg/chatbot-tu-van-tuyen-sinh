import os
import glob
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import gradio as gr
from seed_data import load_data_from_folder, vector_store1,vector_store2




# 4. HÀM TẠO HYBRID RETRIEVER (QUAN TRỌNG)
# ---------------------------------------------------------
def create_hybrid_retriever(vectorstore, chunks):
    """
    Tạo bộ tìm kiếm lai: Kết hợp Keyword (BM25) và Semantic (Vector).
    """
    print("🔍 Đang cấu hình Hybrid Retrieval...")
    
    # 1. Keyword Retriever (BM25) - Tốt cho tìm kiếm tên riêng, mã ngành, con số chính xác
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10  # Lấy top 10 kết quả từ khóa

    # 2. Vector Retriever (Chroma) - Tốt cho tìm kiếm ngữ nghĩa, câu hỏi mơ hồ
    chroma_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 100} # Lấy top 100 kết quả ngữ nghĩa
    )

    # 3. Ensemble (Kết hợp)
    # weights=[0.4, 0.6]: 40% ưu tiên từ khóa, 60% ưu tiên ngữ nghĩa (có thể điều chỉnh)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )
    
    return ensemble_retriever

# 5. HÀM KHỞI TẠO CHATBOT CHAIN
# ---------------------------------------------------------
def setup_rag_chain(retriever):
    """
    Kết nối Retriever với LLM (OpenAI) và bộ nhớ hội thoại.
    """
    # Cấu hình LLM
    # Bạn có thể thay đổi base_url nếu dùng proxy hoặc service khác
    '''llm = ChatOpenAI(
        model_name="gpt-4o-mini", # Hoặc gpt-3.5-turbo
        temperature=0.3,          # Giữ nhiệt độ thấp để câu trả lời chính xác, ít bịa
        streaming=True
    )'''

    llm = ChatOpenAI(
    api_key="sk-bnPOclUlNLW7xF40Xdi35PtWhXA2k8S6gprHkeHGP9XuWQY7",
    base_url="https://gpt1.shupremium.com/v1",
    temperature=0.7, 
    model_name="gpt-4o-mini",
    )


    # Bộ nhớ hội thoại (Nhớ 5 cặp câu hỏi-trả lời gần nhất)
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=True
    )

    system_template = (
    "Bạn là một chuyên gia tư vấn tuyển sinh đại học tại Việt Nam . "
    "Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến điểm tuyển sinh đầu vào, quy chế tuyển sinh, trường học, các phương thức tuyển sinh và cách tính điểm theo cách phương thức tuyển sinh một cách ngắn gọn và chính xác. "
    "Nếu bạn không biết câu trả lời, hãy nói rõ rằng bạn không biết. "
    "Tuyệt đối không bịa ra thông tin nếu không có ngữ cảnh liên quan được cung cấp."
    "Hãy sử dụng thông tin ngữ cảnh dưới đây để trả lời câu hỏi ở cuối:"

    )


    human_template = """Sử dụng thông tin ngữ cảnh sau để trả lời câu hỏi:
    
    Ngữ cảnh (Context):
    {context}

    Câu hỏi (Question): {question}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    # Tạo Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": chat_prompt} 
    )
    
    return qa_chain

# 6. MAIN & GIAO DIỆN
# ---------------------------------------------------------
# Biến toàn cục để lưu chain
global_chain = None

def init_system1():
    global global_chain
    # 1. Load data
    chunks = load_data_from_folder(folders = glob.glob("Data/*"))
    # 2. Vector DB
    vector_db = vector_store1(chunks)
    # 3. Retriever
    hybrid_retriever = create_hybrid_retriever(vector_db, chunks)
    # 4. Setup Chain
    global_chain = setup_rag_chain(hybrid_retriever)
    print("🚀 Hệ thống đã khởi động xong!")

def init_system2():
    global global_chain
    # 1. Load data
    chunks = load_data_from_folder(folders = glob.glob("Data/*"))
    # 2. Vector DB
    vector_db = vector_store2(chunks)
    # 3. Retriever
    hybrid_retriever = create_hybrid_retriever(vector_db, chunks)
    # 4. Setup Chain
    global_chain = setup_rag_chain(hybrid_retriever)

def chat_interface(message, history):
    """Hàm xử lý chat cho Gradio"""
    if global_chain is None:
        return "Hệ thống đang khởi động, vui lòng đợi..."
    
    try:
        response = global_chain.invoke({"question": message})
        return response["answer"]
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}"

# Chạy hệ thống
if __name__ == "__main__":
    print("======================================")
    print("🚀 HỆ THỐNG TRỢ LÝ TUYỂN SINH (Hybrid RAG)")
    print("1. Tạo lại Vector DB từ đầu")
    print("2. Không tạo lại Vector DB (chạy luôn)")
    print("======================================")

    choice = input("👉 Nhập lựa chọn (1 hoặc 2): ").strip()

    if choice == "1":
        print("🔄 Đang tạo lại Vector DB...")
        # Khởi tạo pipeline
        init_system1()
    else:
        print("🔄 Không tạo lại Vector DB (chạy luôn)")
        # Khởi tạo pipeline
        init_system2()
        
    # Khởi chạy giao diện
    print("🌐 Đang mở giao diện Gradio...")
    gr.ChatInterface(
        chat_interface, 
        type="messages",
        title="Trợ lý Tuyển sinh Đại học (Hybrid RAG)",
        description="Hỏi đáp thông tin tuyển sinh sử dụng công nghệ tìm kiếm lai (Từ khóa + Ngữ nghĩa)."
    ).launch()