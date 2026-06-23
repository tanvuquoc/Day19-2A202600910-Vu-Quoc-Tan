import os
import sys
import glob
import time
import json
import networkx as nx
import matplotlib.pyplot as plt

# Cấu hình encoding trên Windows để in tiếng Việt không lỗi
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Khởi tạo mô hình ngôn ngữ (Groq)
LLM_MODEL = "llama-3.3-70b-versatile"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Khởi tạo Embedding model cục bộ miễn phí (do không có OpenAI key)
print("Đang tải mô hình Embedding (HuggingFace)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# BƯỚC 1: ĐỌC DỮ LIỆU & TRÍCH XUẤT THỰC THỂ
# ==========================================
def load_and_extract_dataset(dataset_path, max_files=3):
    print(f"\n--- Bước 1: Trích xuất thực thể và quan hệ từ Dataset ---")
    
    # Đọc danh sách các file txt
    file_pattern = os.path.join(dataset_path, "*.txt")
    files = glob.glob(file_pattern)
    files = sorted(files)[:max_files] # CHỈ LẤY N FILE ĐẦU TIÊN ĐỂ CHẠY NHANH
    
    print(f"Đã tìm thấy {len(files)} files để xử lý (giới hạn {max_files} file để tối ưu thời gian).")
    
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            chunks = text_splitter.split_text(content)
            all_chunks.extend(chunks)
            
    print(f"Tổng cộng đã chia thành {len(all_chunks)} chunks văn bản.")
    
    all_triples = []
    
    # Prompt trích xuất Triples
    prompt_template = """
Hãy trích xuất các thực thể và mối quan hệ giữa chúng từ văn bản dưới đây.
Kết quả trả về PHẢI là một mảng chứa các mảng con 3 phần tử: [Subject, Relation, Object].
Ví dụ: [["Apple", "CEO", "Tim Cook"], ["Tesla", "PRODUCES", "Electric Vehicles"]]
KHÔNG giải thích gì thêm, chỉ trả về đúng định dạng JSON array đó.
Văn bản: "{text}"
Kết quả:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm
    
    print("Đang gọi LLM trích xuất đồ thị (có thể mất vài phút tuỳ theo mạng)...")
    for i, chunk in enumerate(all_chunks):
        try:
            print(f"  Đang xử lý chunk {i+1}/{len(all_chunks)}...")
            response = chain.invoke({"text": chunk})
            content = response.content.replace("```json", "").replace("```", "").strip()
            if content:
                import ast
                try:
                    # Chuyển chuỗi thành list Python
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, list) and len(item) == 3:
                                all_triples.append(item)
                except Exception as ex:
                    print(f"  [Debug] Không thể parse LLM output: {content}")
            time.sleep(1) # Tránh Rate Limit của API miễn phí
        except Exception as e:
            print(f"  [Lỗi parse JSON ở chunk {i+1}]: Bỏ qua...")
            
    # Lọc trùng lặp (Deduplication)
    unique_triples = []
    for t in all_triples:
        if t not in unique_triples and len(t) == 3:
            unique_triples.append(t)
            
    print(f"Hoàn thành trích xuất! Thu được {len(unique_triples)} Triples độc nhất.")
    return all_chunks, unique_triples

# ==========================================
# BƯỚC 2: XÂY DỰNG ĐỒ THỊ (GRAPH CONSTRUCTION)
# ==========================================
def step2_graph_construction(triples):
    print("\n--- Bước 2: Xây dựng Đồ thị tri thức (Knowledge Graph) ---")
    G = nx.DiGraph()
    for subject, relation, obj in triples:
        G.add_edge(subject, obj, label=relation)
            
    print(f"Đồ thị đã được xây dựng với {G.number_of_nodes()} nodes và {G.number_of_edges()} edges.")
    
    # Lưu ảnh đồ thị (Deliverable 2)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=1500, font_size=8, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Tech Company Knowledge Graph")
    plt.savefig("knowledge_graph.png")
    print("Đã lưu ảnh đồ thị vào file 'knowledge_graph.png'")
    
    return G

# ==========================================
# BƯỚC 3 & 4: TRUY VẤN VÀ SO SÁNH (FLAT RAG vs GRAPHRAG)
# ==========================================
def build_flat_rag(chunks):
    print("\n--- Xây dựng Flat RAG (FAISS Vector Store) ---")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def query_flat_rag(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    prompt = PromptTemplate(
        template="Dựa vào các đoạn văn bản sau đây:\n{context}\n\nHãy trả lời câu hỏi: {question}\nNếu không có thông tin, hãy nói 'Không biết'.",
        input_variables=["context", "question"]
    )
    return (prompt | llm).invoke({"context": context, "question": question}).content

def query_graph_rag(question, graph):
    # 1. Tìm thực thể chính trong câu hỏi
    prompt_extract = PromptTemplate(
        template="Trả về đúng 1 thực thể chính quan trọng nhất trong câu hỏi sau. KHÔNG trả lời thêm bất cứ từ nào khác.\nCâu hỏi: {question}\nThực thể:",
        input_variables=["question"]
    )
    main_entity = (prompt_extract | llm).invoke({"question": question}).content.strip()
    
    # 2. Duyệt 2-hop (BFS đơn giản)
    context_triples = []
    if main_entity in graph.nodes:
        for u, v, data in graph.edges(data=True):
            if u == main_entity or v == main_entity:
                context_triples.append(f"{u} {data['label']} {v}")
                neighbor = v if u == main_entity else u
                for u2, v2, data2 in graph.edges(data=True):
                    if (u2 == neighbor or v2 == neighbor) and (u2 != main_entity and v2 != main_entity):
                        context_triples.append(f"{u2} {data2['label']} {v2}")
                        
    context_text = ". ".join(list(set(context_triples)))
    
    # 3. Trả lời dựa trên Graph Context
    prompt_answer = PromptTemplate(
        template="Dựa vào các mối quan hệ đồ thị sau đây (nếu có): {context}\n\nHãy trả lời câu hỏi: {question}\nNếu không có thông tin, hãy nói 'Không biết'.",
        input_variables=["context", "question"]
    )
    return (prompt_answer | llm).invoke({"context": context_text, "question": question}).content

# ==========================================
# CHẠY TOÀN BỘ PIPELINE
# ==========================================
if __name__ == "__main__":
    dataset_folder = r"D:\AI Vingroup\Lab track 3\dataset"
    
    start_time = time.time()
    
    # 1. Trích xuất
    chunks, triples = load_and_extract_dataset(dataset_folder, max_files=3)
    
    # 2. Xây dựng đồ thị
    graph = step2_graph_construction(triples)
    
    # 3. Xây dựng Flat RAG để so sánh
    flat_vectorstore = build_flat_rag(chunks)
    
    # 4. Đánh giá (Evaluation)
    print("\n==============================================")
    print("BƯỚC 4: SO SÁNH FLAT RAG VÀ GRAPHRAG (EVALUATION)")
    print("==============================================")
    
    test_questions = [
        "Liệt kê các công ty được nhắc đến trong dữ liệu.",
        "Mối quan hệ giữa các thực thể trong tài liệu là gì?",
        "Tóm tắt ngắn gọn các chính sách thúc đẩy xe điện ở các thành phố của Mỹ."
    ]
    
    for q in test_questions:
        print(f"\n[Câu hỏi]: {q}")
        print("-" * 50)
        
        flat_ans = query_flat_rag(q, flat_vectorstore)
        print(f"🤖 Flat RAG trả lời:\n{flat_ans}")
        print("-" * 50)
        
        graph_ans = query_graph_rag(q, graph)
        print(f"🕸️ GraphRAG trả lời:\n{graph_ans}")
        print("=" * 50)
        
    end_time = time.time()
    print(f"\n[PHÂN TÍCH CHI PHÍ]: Thời gian chạy toàn bộ pipeline: {end_time - start_time:.2f} giây.")
    print("Số lượng API Calls (ước tính):", len(chunks) + len(test_questions) * 3)
