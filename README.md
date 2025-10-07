# Multi-Agent Medical QA System

Hệ thống multi-agent sử dụng Gemini, LangChain và LangGraph để trả lời các câu hỏi y tế và đánh giá trên các benchmark như MedQA, PubMedQA.

## Workflow

```
Input Question → Coordinator → 
                                    ↓
    ┌───────────┴───────────┐
    ↓                                                             ↓
Web search                                   Reasoning Agent
    ↓                                                             ↓
    └───────────┬───────────┘
                ↓
           Validator
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
│   ├── coordinator.py       # Agent điều phối
│   ├── web_search.py        # Agent tìm kiếm (Tavily + PubMed)
│   ├── reasoning.py         # Agent suy luận
│   ├── validator.py         # Agent kiểm chứng
│   └── answer_generator.py  # Agent tạo câu trả lời
├── workflows/
│   ├── __init__.py
│   └── medical_qa_graph.py  # LangGraph workflow
├── benchmarks/
│   ├── __init__.py
│   ├── medqa_eval.py        # Đánh giá MedQA
│   └── pubmedqa_eval.py     # Đánh giá PubMedQA
├── utils/
│   ├── __init__.py
│   ├── config.py            # Cấu hình
│   └── metrics.py           # Tính toán metrics
├── main.py                  # File chạy chính
├── run_benchmark.py         # Chạy đánh giá
├── requirements.txt
├── .env.example
└── README.md
```

## Cài đặt

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Keys

1. Tạo file `.env` từ template:
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

2. Lấy API Keys:
   - **Google Gemini API**: https://makersuite.google.com/app/apikey
   - **Tavily API**: https://tavily.com/ (miễn phí)

3. Điền vào file `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
TEMPERATURE=0.3

# Tùy chọn: Tùy chỉnh model cho từng agent (xem CONFIG_GUIDE.md)
# REASONING_MODEL=gemini-1.5-pro
# REASONING_TEMPERATURE=0.1
```

### Bước 3: Kiểm tra cài đặt
```bash
python example_usage.py
```

## Sử dụng

### Chạy một câu hỏi đơn lẻ:
```bash
python main.py --question "What is the most common cause of pneumonia?"
```

### Chạy benchmark trên MedQA:
```bash
python run_benchmark.py --dataset medqa --max-samples 100
```

### Chạy benchmark trên PubMedQA:
```bash
python run_benchmark.py --dataset pubmedqa --max-samples 100
```

## Các Agent

1. **Coordinator**: Phân tích câu hỏi và quyết định chiến lược tìm kiếm
2. **Web Search Agent**: Tìm kiếm thông tin từ Tavily và PubMed
3. **Reasoning Agent**: Suy luận logic dựa trên kiến thức
4. **Validator**: Kiểm tra tính nhất quán và độ tin cậy
5. **Answer Generator**: Tổng hợp và tạo câu trả lời cuối cùng

## Metrics

- **Accuracy**: Tỷ lệ câu trả lời đúng
- **F1 Score**: Harmonic mean của Precision và Recall
- **Precision/Recall**: Độ chính xác và độ phủ
- **Response Time**: Thời gian xử lý trung bình
- **Confidence Score**: Độ tin cậy của câu trả lời

## Kiến Trúc

Chi tiết kiến trúc hệ thống xem tại [architecture_diagram.md](architecture_diagram.md)

## Hướng Dẫn

- **Quick Start**: Xem [QUICKSTART.md](QUICKSTART.md) để bắt đầu nhanh
- **Development**: Xem [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) để phát triển và mở rộng

## Kết Quả Mong Đợi

### MedQA Benchmark
- **Baseline**: ~40-50% accuracy (random: 25%)
- **Target**: 60-70% accuracy với multi-agent approach
- **SOTA**: 80%+ (với fine-tuned models)

### PubMedQA Benchmark
- **Baseline**: ~50-60% accuracy (random: 33%)
- **Target**: 70-80% accuracy
- **SOTA**: 85%+ (với domain-specific models)

## Ví Dụ Output

```
Question: What is the most common cause of pneumonia in adults?

Answer: B

Explanation: Streptococcus pneumoniae is the most common bacterial cause 
of community-acquired pneumonia in adults. This is supported by multiple 
clinical studies and guidelines. The web search confirmed this with recent 
medical literature, and the reasoning agent arrived at the same conclusion 
based on epidemiological knowledge.

Confidence: 0.89
Sources: 8
```

## Troubleshooting

Xem [QUICKSTART.md](QUICKSTART.md) section "Troubleshooting"

## Roadmap

- [ ] Thêm support cho hình ảnh y tế (X-ray, CT, MRI)
- [ ] Tích hợp thêm datasets (MedMCQA, MMLU-Medical)
- [ ] Triển khai Self-Consistency và Tree-of-Thought
- [ ] Web UI với Streamlit/Gradio
- [ ] API server với FastAPI
- [ ] Fine-tuning cho medical domain
- [ ] Multi-language support (Tiếng Việt)

## Citation

Nếu sử dụng hệ thống này trong nghiên cứu, vui lòng cite:

```bibtex
@software{medical_qa_multiagent,
  title = {Medical QA Multi-Agent System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/medical-qa-multiagent}
}
```

