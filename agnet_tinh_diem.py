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
        search_kwargs={"k": 100} # Lấy top 10 kết quả ngữ nghĩa
    )

    # 3. Ensemble (Kết hợp)
    # weights=[0.4, 0.6]: 40% ưu tiên từ khóa, 60% ưu tiên ngữ nghĩa (có thể điều chỉnh)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )
    
    return ensemble_retriever

def setup_rag_chain(retriever):
    """
    Kết nối Retriever với LLM (OpenAI) để tính toán điểm và gợi ý trường.
    """
    # Dùng temperature thấp để tính toán chính xác
    llm = ChatOpenAI(
        api_key="sk-bnPOclUlNLW7xF40Xdi35PtWhXA2k8S6gprHkeHGP9XuWQY7",
        base_url="https://gpt1.shupremium.com/v1",
        temperature=0.1, 
        model_name="gpt-4o-mini",
    )

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )

    # --- SYSTEM PROMPT: TƯ DUY LOGIC (CHAIN OF THOUGHT) ---
    system_template = """Bạn là Trợ lý AI chuyên về Tuyển sinh Đại học. Nhiệm vụ của bạn là tính điểm xét tuyển và đánh giá khả năng đỗ của học sinh.

    HÃY THỰC HIỆN SUY LUẬN THEO CÁC BƯỚC SAU (BẮT BUỘC):
    
    Bước 1: Phân tích dữ liệu đầu vào
    - Xác định điểm các môn (Toán, Lý, Hóa, Văn, Anh...) và chứng chỉ (IELTS, TOEIC...) từ câu hỏi của người dùng.
    
    Bước 2: Tìm kiếm thông tin trong Ngữ cảnh (Context)
    - Tìm tên trường đại học mà người dùng quan tâm (hoặc tất cả các trường có trong ngữ cảnh).
    - Tìm "Bảng quy đổi điểm IELTS" của trường đó (nếu có IELTS).
    - Tìm "Công thức tính điểm" của trường đó.
    - Tìm "Điểm chuẩn" hoặc "Điểm trúng tuyển" các năm trước.

    Bước 3: Thực hiện tính toán (Hiển thị chi tiết từng phép tính)
    - Nếu có IELTS: Quy đổi IELTS sang điểm thi theo quy chế tìm được ở Bước 2.
    - Áp dụng công thức: Thay số vào để tính Tổng điểm xét tuyển.
    
    Bước 4: Đưa ra đề nghị và Kết luận
    - So sánh Tổng điểm vừa tính với Điểm chuẩn trong ngữ cảnh.
    - Đưa ra nhận định: "Khả năng đỗ Cao/Thấp/An toàn".
    - Gợi ý ngành học phù hợp với số điểm đó.

    LƯU Ý QUAN TRỌNG:
    - Nếu không tìm thấy công thức tính hoặc bảng quy đổi trong ngữ cảnh, hãy trả lời trung thực: "Xin lỗi, dữ liệu hiện tại chưa cập nhật cách tính điểm cho trường này".
    - Tuyệt đối KHÔNG tự bịa ra công thức tính điểm."""


    

    # --- HUMAN PROMPT ---
    human_template = """
    Dữ liệu tham khảo (Context):
    {context}

    Câu hỏi/Hồ sơ của học sinh (Question): 
    {question}
    
    Hãy tính toán và tư vấn chi tiết:
    """

    # Tạo Prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    # Tạo Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True, # Bật lên để xem log tính toán
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