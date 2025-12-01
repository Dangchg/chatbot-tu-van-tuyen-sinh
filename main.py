from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 1. KHỞI TẠO LLM CHO ROUTER (Nên dùng model nhanh, rẻ)
router_llm = ChatOpenAI(
    api_key="sk-bnPOclUlNLW7xF40Xdi35PtWhXA2k8S6gprHkeHGP9XuWQY7",
    base_url="https://gpt1.shupremium.com/v1",
    temperature=0, # Bắt buộc bằng 0 để phân loại chính xác
    model_name="gpt-4o-mini"
)

# 2. HÀM PHÂN LOẠI YÊU CẦU (THE ROUTER)
def classify_intent(question):
    """
    Phân tích câu hỏi để xác định người dùng muốn gì.
    Output: 'CALCULATION', 'ADVISORY', hoặc 'INFO'
    """
    system_instruction = """
    Bạn là một bộ định tuyến (Router) thông minh. Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào 1 trong 3 nhóm sau:

    1. 'CALCULATION': Nếu người dùng cung cấp điểm số chi tiết các môn (Toán, Lý, Anh, IELTS...) và yêu cầu tính điểm xét tuyển.
       - Ví dụ: "IELTS 7.0 toán 8 lý 9 thì bao nhiêu điểm?", "Tính điểm giúp mình với các điểm sau..."

    2. 'ADVISORY': Nếu người dùng cung cấp TỔNG ĐIỂM (hoặc điểm áng chừng) và hỏi có thể đỗ trường nào/ngành nào.
       - Ví dụ: "Mình được 24 điểm nên vào trường nào?", "25 điểm có đỗ Kinh tế quốc dân không?", "Tư vấn chọn trường khối A".

    3. 'INFO': Các câu hỏi thông tin chung, quy chế, học phí, lịch sử, ký túc xá... không liên quan đến tính toán cụ thể.
       - Ví dụ: "Học phí Bách Khoa là bao nhiêu?", "Trường có mấy cơ sở?", "Quy chế tuyển thẳng thế nào?".

    CHỈ TRẢ VỀ DUY NHẤT 1 TỪ KHÓA: CALCULATION, ADVISORY, HOẶC INFO. KHÔNG GIẢI THÍCH GÌ THÊM.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{question}")
    ])

    # Tạo chain phân loại
    route_chain = prompt | router_llm | StrOutputParser()
    
    # Thực thi
    category = route_chain.invoke({"question": question})
    return category.strip().upper()

# 3. CẤU HÌNH 3 AGENT (Tái sử dụng các prompt bạn đã làm)
# --------------------------------------------------------

def get_calculation_chain(retriever, llm, memory):
    # Copy code phần Prompt tính toán (IELTS + Toán...) vào đây
    system_template = "Bạn là máy tính tuyển sinh. Hãy trích xuất điểm và tính toán chi tiết..."
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", "{context}\n\n{question}")])
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory, combine_docs_chain_kwargs={"prompt": prompt})

def get_advisory_chain(retriever, llm, memory):
    # Copy code phần Prompt tư vấn ngược (24 điểm đỗ trường nào) vào đây
    system_template = "Bạn là chuyên gia tư vấn. Hãy lọc các trường phù hợp với mức điểm tổng..."
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", "{context}\n\n{question}")])
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory, combine_docs_chain_kwargs={"prompt": prompt})

def get_general_info_chain(retriever, llm, memory):
    # Copy code phần Prompt hỏi đáp thông thường vào đây
    system_template = "Bạn là trợ lý ảo. Hãy trả lời thông tin dựa trên ngữ cảnh..."
    prompt = ChatPromptTemplate.from_messages([("system", system_template), ("human", "{context}\n\n{question}")])
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory, combine_docs_chain_kwargs={"prompt": prompt})

# 4. HÀM XỬ LÝ CHÍNH (MAIN HANDLER)
# --------------------------------------------------------
def main_chat_handler(message, history, retriever, memory):
    """
    Đây là hàm sẽ được gọi bởi Gradio
    """
    print(f"📥 Câu hỏi nhận được: {message}")
    
    # BƯỚC 1: ĐỊNH TUYẾN
    intent = classify_intent(message)
    print(f"🔀 Router quyết định chuyển hướng sang: {intent}")

    # Cấu hình LLM chung cho các Agent con
    llm_worker = ChatOpenAI(
        api_key="sk-bnPOclUlNLW7xF40Xdi35PtWhXA2k8S6gprHkeHGP9XuWQY7",
        base_url="https://gpt1.shupremium.com/v1",
        temperature=0.1, 
        model_name="gpt-4o-mini"
    )

    # BƯỚC 2: CHỌN CHAIN PHÙ HỢP
    if intent == "CALCULATION":
        active_chain = get_calculation_chain(retriever, llm_worker, memory)
        prefix = "🧮 [Chế độ Tính Điểm]: " # (Optional) Để debug xem đúng ko
    elif intent == "ADVISORY":
        active_chain = get_advisory_chain(retriever, llm_worker, memory)
        prefix = "🎓 [Chế độ Tư Vấn Chọn Trường]: "
    else:
        active_chain = get_general_info_chain(retriever, llm_worker, memory)
        prefix = "ℹ️ [Chế độ Thông Tin]: "

    # BƯỚC 3: TRẢ LỜI
    try:
        response = active_chain.invoke({"question": message})
        # return prefix + response["answer"] # Có thể bỏ prefix nếu muốn tự nhiên
        return response["answer"]
    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"

# 5. CẬP NHẬT GRADIO
# Trong phần main, bạn chỉ cần gọi main_chat_handler