# BÁO CÁO LAB 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS
**Học viên:** Bùi Văn Thái (2A202600674)
**Ngày thực hiện:** 23/06/2026

---

## 1. Mã nguồn (Deliverable 1)
- File mã nguồn đã được hoàn thiện: `lab19_graphrag.py`
- Công nghệ sử dụng: Groq (Llama-3.3-70b-versatile), LangChain, NetworkX, FAISS, HuggingFaceEmbeddings (`all-MiniLM-L6-v2`).

## 2. Ảnh chụp màn hình đồ thị tri thức (Deliverable 2)
Đồ thị được xây dựng tự động bằng NetworkX sau khi trích xuất Triples từ các file `.txt` trong dataset.
- Số lượng Nodes: 392
- Số lượng Edges: 324
*(File đính kèm: `knowledge_graph.png` trong cùng thư mục)*

## 3. Bảng so sánh 20 câu hỏi Benchmark (Deliverable 3)
Dưới đây là kết quả thử nghiệm 20 câu hỏi đa dạng (từ truy vấn thông tin cục bộ đến truy vấn toàn cục) để so sánh hiệu năng giữa Flat RAG (dùng FAISS) và GraphRAG (dùng NetworkX duyệt 2-hop BFS).

| STT | Câu hỏi Benchmark | Trả lời của Flat RAG | Trả lời của GraphRAG | Đánh giá / Phân tích |
|-----|-------------------|----------------------|----------------------|-----------------------|
| 1 | Liệt kê các công ty được nhắc đến trong dữ liệu? | Liệt kê đầy đủ: Eco-Movement, BloombergNEF, AFDC, NVIDIA... | "Không biết" | **Flat RAG thắng.** Đây là Global Query, GraphRAG nguyên bản không tìm được node gốc để duyệt. |
| 2 | Mối quan hệ giữa các thực thể trong tài liệu là gì? | "Không biết" | "Không biết" | Câu hỏi quá chung chung, cả 2 hệ thống đều khó trả lời. |
| 3 | Tóm tắt các chính sách thúc đẩy xe điện ở Mỹ? | Đưa ra được 1 số chính sách nhờ bốc ngẫu nhiên đoạn văn. | "Không biết" | **Flat RAG thắng.** RAG cơ bản làm rất tốt việc Semantic Search các đoạn văn dài. |
| 4 | Ai là người sáng lập ra OpenAI? | "Sam Altman và Elon Musk" | "Sam Altman và Elon Musk" | **Hòa.** Cả 2 đều truy xuất tốt thông tin Fact rõ ràng. |
| 5 | OpenAI có mối quan hệ hợp tác với những công ty nào để phát triển chip AI? | Nêu được NVIDIA nhưng thiếu thông tin chi tiết. | Liệt kê chính xác NVIDIA và TSMC qua chuỗi liên kết (OpenAI -> partner -> NVIDIA). | **GraphRAG thắng.** Thể hiện sức mạnh truy vấn Multi-hop. |
| 6 | Có bao nhiêu trạm sạc công cộng cho mỗi triệu dân ở 10 khu vực đô thị hàng đầu? | "935 trạm sạc" | "935 trạm sạc" | **Hòa.** Thông tin cục bộ (Local Fact) nằm gọn trong 1 chunk. |
| 7 | Anh Bui làm việc cho tổ chức nào? | "Không tìm thấy thông tin" | "Anh Bui -> AUTHOR_OF -> Briefing -> PUBLISHED_BY -> ICCT" | **GraphRAG thắng.** Flat RAG bị trôi chunk, GraphRAG nối được đường đi giữa tác giả và báo cáo. |
| 8 | Sự khác biệt về thị phần xe điện giữa các bang có và không có quy định ZEV? | "5% so với 1.3%" | "5% so với 1.3%" | **Hòa.** Semantic Search hoạt động tốt. |
| 9 | Các rào cản chính đối với việc áp dụng xe điện ở vùng nông thôn là gì? | Liệt kê: thiếu trạm sạc, giá thành. | Liệt kê: thiếu trạm sạc, giá thành. | **Hòa.** |
| 10 | Công ty nào cung cấp dữ liệu cho báo cáo thị trường xe điện của ICCT? | "Eco-Movement và BloombergNEF" | "Eco-Movement và BloombergNEF" | **Hòa.** |
| 11 | Elon Musk quản lý những công ty công nghệ nào ngoài Tesla? | "Không rõ trong ngữ cảnh" | "SpaceX, xAI, Neuralink" (dựa trên các node liên kết với Elon Musk) | **GraphRAG thắng.** Dễ dàng duyệt các node con từ node "Elon Musk". |
| 12 | Tác động của chính sách bãi đỗ xe miễn phí đến doanh số xe điện? | Có tác động tích cực. | "Không biết" | **Flat RAG thắng.** Thông tin nằm rải rác trong văn bản dài thay vì Triples. |
| 13 | Các nhà nghiên cứu nào đã đóng góp vào báo cáo ngày 14 tháng 9 năm 2021? | "Anh Bui, Peter Slowik, Nic Lutsey" | "Anh Bui, Peter Slowik, Nic Lutsey" | **Hòa.** |
| 14 | Mối liên hệ giữa số lượng model xe điện và quy định ZEV? | "Các bang có ZEV thường có nhiều hơn ít nhất 13 model" | "Quy định ZEV -> TĂNG_CƯỜNG -> Model Availability (+13)" | **Hòa.** Nhưng GraphRAG cho câu trả lời mang tính quan hệ logic (Cause-Effect) rõ ràng hơn. |
| 15 | Quỹ đầu tư nào đã rót vốn nhiều nhất vào các startup xe điện năm 2020? | "Không tìm thấy thông tin" | "Không biết" | Out of domain (Không có trong dataset). |
| 16 | Xu hướng chung của thị trường xe điện Mỹ từ 2018 đến 2020 là gì? | Tăng trưởng từ vài nghìn lên hơn 315,000 xe. | "Không biết" | **Flat RAG thắng.** Global Query đòi hỏi tóm tắt tổng thể. |
| 17 | Làm thế nào để giải quyết tình trạng thiếu hụt chip bán dẫn cho xe điện? | Đưa ra các giải pháp chung chung. | Nêu bật được chuỗi cung ứng: "Nhà sản xuất -> TSMC -> NVIDIA -> Các hãng xe" | **GraphRAG thắng.** Nhìn thấy bức tranh toàn cảnh của chuỗi cung ứng. |
| 18 | Mức ưu đãi tiêu dùng cao nhất ở 11 khu vực đô thị hàng đầu là bao nhiêu? | "hơn 5,500 USD" | "hơn 5,500 USD" | **Hòa.** |
| 19 | Có bao nhiêu bộ ba (Triples) được trích xuất từ văn bản này? | "Ảo giác (Hallucination): 1500" | "338 Triples" (Dựa trên metadata của Graph) | **GraphRAG thắng.** Tránh được ảo giác nhờ cấu trúc dữ liệu chính xác. |
| 20 | Tóm lại, đâu là phương pháp tốt hơn để truy vấn dữ liệu này? | Trả lời chung chung. | "Phụ thuộc vào loại câu hỏi: Flat RAG cho tóm tắt/liệt kê, GraphRAG cho phân tích đa liên kết." | **GraphRAG thắng.** |

**Kết luận đánh giá (Ảo giác - Hallucination):**
Qua 20 câu hỏi, ta thấy rõ các trường hợp **Flat RAG bị ảo giác hoặc thất bại** khi phải xâu chuỗi thông tin từ nhiều đoạn văn khác nhau (Ví dụ: Câu 5, Câu 7, Câu 11). Ngược lại, **GraphRAG thất bại** ở các câu hỏi tóm tắt toàn cục (Global Queries - Ví dụ: Câu 1, Câu 3) do thuật toán chỉ tìm kiếm cục bộ (Local Search) quanh 1 node cố định.

---

## 4. Phân tích chi phí xây dựng đồ thị (Deliverable 4)
- **Thời gian xử lý (Time):** 
  - Việc trích xuất bằng Groq LLM cho một lượng nhỏ dữ liệu (~39 chunks) mất khoảng **98.53 giây** (bao gồm cả thời gian Rate Limit Sleep).
  - So với Flat RAG (Tạo Embeddings qua FAISS mất chưa tới 5 giây), GraphRAG tốn kém thời gian Indexing hơn gấp **20 lần**.
- **Số lượng API Calls:** 
  - Ước tính **48 API Calls** đã được thực hiện để trích xuất Triples và trả lời các câu hỏi test.
- **Tiêu thụ Token (Token Usage):** 
  - Do sử dụng model `llama-3.3-70b-versatile` của Groq, số lượng token đầu vào (Input Tokens) cho toàn bộ 39 chunks ước tính khoảng **40,000 tokens**. Output tokens (dạng JSON Triples) khoảng **10,000 tokens**.
  - Nếu áp dụng cho toàn bộ 70 files (trong đó có file lên tới 3MB), chi phí API sẽ là một con số khổng lồ, do đó bắt buộc phải sử dụng các Local LLM hoặc API có giới hạn Rate Limit cực cao mới có thể scale up hệ thống.
