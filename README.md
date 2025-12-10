# Multi-Agent Medical QA System with Medprompt

Hệ thống multi-agent sử dụng Gemini, LangChain và LangGraph để trả lời các câu hỏi y tế và đánh giá trên các benchmark như MedQA, PubMedQA.

**Tích hợp Medprompt** - phương pháp prompt engineering tiên tiến từ Microsoft để cải thiện hiệu suất trên các bài toán y tế.

## 🚀 Super Graph - NEW!

**Master Coordinator Agent** với intelligent routing tự động:
- ✅ **Direct Answer**: Trả lời nhanh câu hỏi đơn giản (1-3 giây)
- 🔬 **Medical QA Subgraph**: Phân tích sâu câu hỏi phức tạp (10-30 giây)
- 🖼️ **Image QA Subgraph**: Xử lý ảnh y tế và VQA (5-15 giây)

**Tự động phát hiện độ phức tạp** và route đến workflow phù hợp!

👉 Xem hướng dẫn chi tiết: [SUPER_GRAPH_GUIDE.md](SUPER_GRAPH_GUIDE.md)

## 💬 Multi-turn Chat - NEW!

**Conversation Memory Management** với automatic summarization:
- 🔄 **Multi-turn Conversations**: Duy trì context qua nhiều lượt hội thoại
- 📝 **Auto Summarization**: Tự động tóm tắt conversation khi vượt ngưỡng
- 🧠 **Smart Context**: Kết hợp summary + recent turns cho context tối ưu
- 💾 **Session Management**: Track và export conversation sessions

👉 Xem hướng dẫn chi tiết: [MULTI_TURN_GUIDE.md](MULTI_TURN_GUIDE.md)

## 🌟 Tính năng chính

### 1. Dynamic Few-shot Selection
- Tự động tìm các câu hỏi tương tự từ training set
- Sử dụng embedding model y tế (PubMedBERT)
- K-NN retrieval để chọn examples phù hợp nhất

### 2. Self-Generated Chain-of-Thought (CoT)
- Tạo chuỗi suy luận chi tiết
- Học từ examples tương tự
- Phân tích từng option một cách logic

### 3. Choice Shuffling Ensemble
- Giảm bias vị trí trong câu hỏi trắc nghiệm
- Chạy nhiều variants với options được shuffle
- Majority voting để chọn đáp án cuối cùng

### 4. Self-Consistency (Multiple Sampling)
- Chạy reasoning nhiều lần với temperature cao
- Aggregation qua voting để tăng độ tin cậy
- Phù hợp cho high-stakes questions

### 5. Self-Correction (Reflexion)
- Agent tự phê bình và đánh giá câu trả lời
- Phát hiện lỗ hổng logic và thiếu sót
- Tự động sửa và cải thiện đáp án
- 3 phases: Critique → Correction → Verification

### 6. 🆕 Multimodal Perception (Image Analysis & VQA)
- Phân tích ảnh y tế (X-ray, CT, MRI, đơn thuốc...)
- Visual Question Answering (VQA) trên ảnh y tế
- Hỗ trợ input từ file path hoặc URL
- Tự động routing giữa text workflow và image workflow

## Workflow

### Text-based QA Workflow
```
            Input Question 
                 ↓
      [Coordinator Agent]
         ├── Phân tích câu hỏi
         └── 📌 Dynamic Few-shot Selection (K-NN)
                 ↓
 ┌───────────────┴───────────────┐
 ↓                               ↓
Web Search              [Reasoning Agent]
                            ├── 📌 Self-Generated CoT
                            └── 📌 Self-Consistency (optional)
 ↓                               ↓
 └───────────────┬───────────────┘
                 ↓
          [Validator Agent]
              └── 📌 Choice Shuffling Ensemble
                 ↓
         [Answer Generator]
                 ↓
        [Reflexion Agent]
            ├── Critique (đánh giá)
            ├── Correction (sửa lỗi)
            └── Verification (xác nhận)
                 ↓
              Output
```

### 🆕 Image-based QA Workflow
```
      Input (Image + Question)
                 ↓
         [Image Agent] 🖼️
            ├── Analyze medical image
            └── Extract findings
                 ↓
        [Image Reasoning]
            ├── VQA mode (if question)
            └── Analysis mode (no question)
                 ↓
        [Image Validator]
                 ↓
        [Answer Generator]
                 ↓
              Output
```

## Cấu trúc thư mục

```
DACN/
├── agents/
│   ├── __init__.py
│   ├── coordinator.py       # + Dynamic Few-shot Selection
│   ├── web_search.py        # Tavily + PubMed search
│   ├── reasoning.py         # + Self-Generated CoT
│   ├── validator.py         # + Choice Shuffling Ensemble
│   ├── answer_generator.py  
│   ├── reflexion.py         # Self-Correction (Reflexion)
│   └── image_agent.py       # 🆕 Image Analysis & VQA
├── workflows/
│   ├── __init__.py
│   ├── medical_qa_graph.py  # LangGraph workflow với Medprompt + Reflexion
│   └── image_qa_graph.py    # 🆕 Image QA workflow (subgraph)
├── benchmarks/
│   ├── __init__.py
│   ├── medqa_eval.py        
│   └── pubmedqa_eval.py     
├── utils/
│   ├── __init__.py
│   ├── config.py            # Cấu hình + Medprompt + Reflexion settings
│   ├── metrics.py           
│   ├── embedding_service.py # Vector embeddings
│   ├── knn_retriever.py     # K-NN retrieval
│   └── ensemble.py          # Voting mechanisms
├── data/
│   └── knowledge_base/      # Embedded training examples
├── build_knowledge_base.py  # Script build index
├── run_benchmark.py         # + Medprompt options
├── example_usage.py         
├── .env.example             # 🆕 Template cấu hình
├── requirements.txt
└── README.md
```

## Cài đặt

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Keys

1. Copy file `.env.example` thành `.env`:
```bash
cp .env.example .env
```

2. Điền API keys và cấu hình:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
TEMPERATURE=0.3

# Medprompt settings
ENABLE_FEW_SHOT=true
ENABLE_COT=true
ENABLE_ENSEMBLE=true
ENABLE_SELF_CONSISTENCY=false
SELF_CONSISTENCY_SAMPLES=3

# Reflexion settings
ENABLE_REFLEXION=true
REFLEXION_MAX_ITERATIONS=2
REFLEXION_CONFIDENCE_THRESHOLD=0.7
```

3. Lấy API Keys:
   - **Google Gemini API**: https://makersuite.google.com/app/apikey
   - **Tavily API**: https://tavily.com/

### Bước 3: Build Knowledge Base (cho Few-shot Selection)
```bash
python build_knowledge_base.py --train_file MedQA/4_options/phrases_no_exclude_train.jsonl
```

### Bước 4: Kiểm tra cài đặt
```bash
python example_usage.py
```

## Sử dụng

### 🆕 Super Graph (Intelligent Routing - Recommended)

Super Graph tự động phát hiện độ phức tạp và route đến workflow phù hợp:

```bash
# Câu hỏi đơn giản (trả lời trực tiếp, nhanh)
python main.py --question "What is hypertension?"

# Câu hỏi phức tạp (route đến Medical QA subgraph)
python main.py --question "A 45-year-old man presents with chest pain..." \
  --options "A. Anterior MI" "B. Inferior MI" "C. PE" "D. Dissection"

# Ảnh y tế (route đến Image QA subgraph)
python main.py --image "path/to/xray.jpg" --question "What is the diagnosis?"
```

**Xem ví dụ chi tiết**: `python example_super_graph.py`

### Legacy Mode (Direct Routing)

Nếu muốn bỏ qua Super Graph và dùng routing trực tiếp:

```bash
# Chạy một câu hỏi đơn lẻ (Text):
python main.py --legacy-mode --question "What is the most common cause of pneumonia?" \
  --options "A. Virus" "B. Bacteria" "C. Fungus" "D. Parasite"
```

### 🆕 Phân tích ảnh y tế:
```bash
# Phân tích ảnh từ file
python main.py --image "path/to/chest_xray.jpg"

# Phân tích ảnh từ URL
python main.py --image "https://example.com/medical-image.png"

# VQA - Trả lời câu hỏi về ảnh
python main.py --image "path/to/xray.jpg" \
  --question "Is there any sign of pneumonia?"

# VQA với multiple choice
python main.py --image "path/to/xray.jpg" \
  --question "What type of imaging is shown?" \
  --options "A. MRI" "B. CT scan" "C. X-ray" "D. Ultrasound"
```

### Chạy với Reflexion (Self-Correction):
```bash
python main.py --reflexion --question "..." --options "A. ..." "B. ..."
```

### Chạy KHÔNG có Reflexion:
```bash
python main.py --no-reflexion --question "..." --options "A. ..." "B. ..."
```

### Chạy benchmark với Medprompt:
```bash
python run_benchmark.py --dataset medqa --max-samples 100
```

### Chạy benchmark KHÔNG có Medprompt (để so sánh):
```bash
python run_benchmark.py --dataset medqa --max-samples 100 --no-medprompt
```

### Tùy chọn Medprompt:
```bash
# Disable từng feature
python run_benchmark.py --no-few-shot           # Không dùng few-shot
python run_benchmark.py --no-cot                # Không dùng CoT
python run_benchmark.py --no-ensemble           # Không dùng ensemble

# Bật Self-Consistency (cho high-stakes questions)
python run_benchmark.py --self-consistency --sc-samples 5

# Tùy chỉnh parameters
python run_benchmark.py --few-shot-k 5 --ensemble-variants 7
```

## Cấu hình Medprompt

Xem chi tiết tại:
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Cấu hình tổng hợp
- [MEDPROMPT_GUIDE.md](MEDPROMPT_GUIDE.md) - Hướng dẫn Medprompt chi tiết

### Quick Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_FEW_SHOT` | true | Bật few-shot selection |
| `FEW_SHOT_K` | 3 | Số examples tương tự |
| `ENABLE_COT` | true | Bật Chain-of-Thought |
| `ENABLE_ENSEMBLE` | true | Bật choice shuffling |
| `ENSEMBLE_VARIANTS` | 5 | Số variants |
| `ENABLE_SELF_CONSISTENCY` | false | Bật self-consistency (multiple sampling) |
| `SELF_CONSISTENCY_SAMPLES` | 3 | Số lần sampling |
| `ENABLE_REFLEXION` | true | Bật self-correction (Reflexion) |
| `REFLEXION_MAX_ITERATIONS` | 2 | Số vòng lặp sửa lỗi tối đa |
| `IMAGE_MODEL` | gemini-2.5-flash | 🆕 Model cho image analysis |
| `IMAGE_TEMPERATURE` | 0.3 | 🆕 Temperature cho image agent |

## Các Agent

### Text-based Agents
1. **Coordinator**: Phân tích câu hỏi + **Dynamic Few-shot Selection**
2. **Web Search Agent**: Tìm kiếm từ Tavily và PubMed
3. **Reasoning Agent**: Suy luận logic + **Self-Generated CoT** + **Self-Consistency**
4. **Validator**: Kiểm tra tính nhất quán + **Choice Shuffling Ensemble**
5. **Answer Generator**: Tổng hợp câu trả lời cuối cùng (Structured Output với Pydantic)
6. **Reflexion Agent**: Tự phê bình và sửa lỗi câu trả lời
   - **Critique**: Đánh giá logic, accuracy, evidence
   - **Correction**: Sửa và cải thiện câu trả lời
   - **Verification**: Xác nhận correction tốt hơn original

### 🆕 Multimodal Agent
7. **Image Agent**: Phân tích ảnh y tế và VQA
   - **analyze_image()**: Phân tích tổng quan (findings, interpretation)
   - **answer_question()**: Trả lời câu hỏi dựa trên ảnh
   - Hỗ trợ: X-ray, CT, MRI, đơn thuốc, lab results...

## Metrics

- **Accuracy**: Tỷ lệ câu trả lời đúng
- **F1 Score**: Harmonic mean của Precision và Recall
- **Precision/Recall**: Độ chính xác và độ phủ
- **Response Time**: Thời gian xử lý trung bình
- **Confidence Score**: Độ tin cậy của câu trả lời
- **Ensemble Consistency**: Độ nhất quán giữa các predictions (mới)

## Ví Dụ Output

```
Question: What is the most common cause of pneumonia in adults?

Answer: B

Explanation: Streptococcus pneumoniae is the most common bacterial cause 
of community-acquired pneumonia in adults.

Confidence: 0.92
Sources: 8
Time taken: 45.32 seconds

--- Medprompt Info ---
Few-shot examples used: 3
CoT reasoning: True
Ensemble used: True
Ensemble consistency: 0.80

--- Reflexion (Self-Correction) Info ---
Performed: True
Iterations: 1
Original answer: C
Original confidence: 0.65
Correction applied: Yes
Reason: Improved reasoning after critique
```

## Tài liệu

- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Hướng dẫn cấu hình
- [MEDPROMPT_GUIDE.md](MEDPROMPT_GUIDE.md) - Hướng dẫn Medprompt
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Hướng dẫn phát triển
- [architecture_diagram.md](architecture_diagram.md) - Kiến trúc chi tiết

## Roadmap

- [x] ~~Triển khai Medprompt (Few-shot, CoT, Ensemble)~~
- [x] ~~Self-Consistency (Multiple Sampling)~~
- [x] ~~Structured Output với Pydantic Parser~~
- [x] ~~Self-Correction với Reflexion~~
- [x] ~~Multimodal Perception (Image Analysis & VQA)~~ 🆕
- [ ] Tích hợp thêm datasets (MedMCQA, MMLU-Medical)
- [ ] Image-based benchmark evaluation
- [ ] Web UI với Streamlit/Gradio
- [ ] API server với FastAPI

## References

- [Medprompt Paper](https://arxiv.org/abs/2311.16452) - Microsoft Research
- [MedQA Dataset](https://github.com/jind11/MedQA)
- [Sentence Transformers](https://www.sbert.net/)
