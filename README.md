# Multi-Agent Medical QA System with Medprompt

Hệ thống multi-agent sử dụng Gemini, LangChain và LangGraph để trả lời các câu hỏi y tế và đánh giá trên các benchmark như MedQA, PubMedQA.

**Tích hợp Medprompt** - phương pháp prompt engineering tiên tiến từ Microsoft để cải thiện hiệu suất trên các bài toán y tế.

## 🌟 Tính năng mới: Medprompt Integration

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

## Workflow

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
                            └── 📌 Self-Generated CoT
 ↓                               ↓
 └───────────────┬───────────────┘
                 ↓
          [Validator Agent]
              └── 📌 Choice Shuffling Ensemble
                 ↓
         Answer Generator
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
│   └── answer_generator.py  
├── workflows/
│   ├── __init__.py
│   └── medical_qa_graph.py  # LangGraph workflow với Medprompt
├── benchmarks/
│   ├── __init__.py
│   ├── medqa_eval.py        
│   └── pubmedqa_eval.py     
├── utils/
│   ├── __init__.py
│   ├── config.py            # Cấu hình + Medprompt settings
│   ├── metrics.py           
│   ├── embedding_service.py # 🆕 Vector embeddings
│   ├── knn_retriever.py     # 🆕 K-NN retrieval
│   └── ensemble.py          # 🆕 Voting mechanisms
├── data/
│   └── knowledge_base/      # 🆕 Embedded training examples
├── build_knowledge_base.py  # 🆕 Script build index
├── run_benchmark.py         # + Medprompt options
├── example_usage.py         
├── requirements.txt
└── README.md
```

## Cài đặt

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Keys

1. Tạo file `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
TEMPERATURE=0.3

# Medprompt settings
ENABLE_FEW_SHOT=true
ENABLE_COT=true
ENABLE_ENSEMBLE=true
```

2. Lấy API Keys:
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

### Chạy một câu hỏi đơn lẻ:
```bash
python main.py --question "What is the most common cause of pneumonia?"
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
python run_benchmark.py --no-few-shot    # Không dùng few-shot
python run_benchmark.py --no-cot         # Không dùng CoT
python run_benchmark.py --no-ensemble    # Không dùng ensemble

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

## Các Agent

1. **Coordinator**: Phân tích câu hỏi + **Dynamic Few-shot Selection**
2. **Web Search Agent**: Tìm kiếm từ Tavily và PubMed
3. **Reasoning Agent**: Suy luận logic + **Self-Generated CoT**
4. **Validator**: Kiểm tra tính nhất quán + **Choice Shuffling Ensemble**
5. **Answer Generator**: Tổng hợp câu trả lời cuối cùng

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

Confidence: 0.89
Sources: 8

--- Medprompt Info ---
Few-shot examples used: 3
CoT reasoning: True
Ensemble used: True
Ensemble consistency: 0.80
Predictions: ['B', 'B', 'B', 'B', 'A']
Vote distribution: {'B': 0.8, 'A': 0.2}
```

## Tài liệu

- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Hướng dẫn cấu hình
- [MEDPROMPT_GUIDE.md](MEDPROMPT_GUIDE.md) - Hướng dẫn Medprompt
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Hướng dẫn phát triển
- [architecture_diagram.md](architecture_diagram.md) - Kiến trúc chi tiết

## Roadmap

- [x] ~~Triển khai Medprompt (Few-shot, CoT, Ensemble)~~
- [ ] Thêm support cho hình ảnh y tế (X-ray, CT, MRI)
- [ ] Tích hợp thêm datasets (MedMCQA, MMLU-Medical)
- [ ] Web UI với Streamlit/Gradio
- [ ] API server với FastAPI

## References

- [Medprompt Paper](https://arxiv.org/abs/2311.16452) - Microsoft Research
- [MedQA Dataset](https://github.com/jind11/MedQA)
- [Sentence Transformers](https://www.sbert.net/)
