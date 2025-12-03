"""
Script to build the knowledge base by embedding training questions.
This creates a searchable index for dynamic few-shot selection.

Usage:
    python build_knowledge_base.py --train_file MedQA/4_options/phrases_no_exclude_train.jsonl
    python build_knowledge_base.py --train_file MedQA/train.jsonl --max_examples 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.embedding_service import EmbeddingService
from utils.knn_retriever import KNNRetriever, load_training_examples


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build knowledge base from training data"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="MedQA/4_options/phrases_no_exclude_train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/knowledge_base/medqa",
        help="Directory to save the index (default: data/knowledge_base/medqa)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (None for all)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        help="HuggingFace model for embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--include_options",
        action="store_true",
        default=True,
        help="Include options in embedding"
    )
    parser.add_argument(
        "--generate_cot",
        action="store_true",
        default=False,
        help="Generate Chain-of-Thought reasoning for examples (requires LLM API)"
    )
    
    return parser.parse_args()


def load_jsonl(filepath: str, max_examples: int = None) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading examples")):
            if max_examples and i >= max_examples:
                break
            
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    
    print(f"Loaded {len(examples)} examples")
    return examples


def prepare_example_with_metadata(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare an example with additional metadata for the knowledge base.
    """
    prepared = {
        'question': example.get('question', ''),
        'answer': example.get('answer', ''),
        'answer_idx': example.get('answer_idx', ''),
        'options': example.get('options', {}),
        'meta_info': example.get('meta_info', ''),
        'metamap_phrases': example.get('metamap_phrases', []),
    }
    
    # Add CoT reasoning if available
    if 'cot_reasoning' in example:
        prepared['cot_reasoning'] = example['cot_reasoning']
    elif 'reasoning' in example:
        prepared['cot_reasoning'] = example['reasoning']
    
    return prepared


def generate_cot_reasoning(
    examples: List[Dict[str, Any]], 
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate Chain-of-Thought reasoning for examples using LLM.
    This is optional and can be run separately.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from utils.config import Config
        
        llm = ChatGoogleGenerativeAI(**Config.get_llm_config('reasoning'))
    except Exception as e:
        print(f"Cannot initialize LLM for CoT generation: {e}")
        return examples
    
    print("Generating Chain-of-Thought reasoning...")
    
    cot_prompt = """Given this medical question, provide step-by-step reasoning to arrive at the answer.

Question: {question}

Options:
{options}

Correct Answer: {answer}

Provide a detailed reasoning process:
1. Identify the key clinical findings
2. Consider the differential diagnosis
3. Apply medical knowledge
4. Explain why the correct answer is right and why other options are wrong

Reasoning:"""
    
    for i, example in enumerate(tqdm(examples, desc="Generating CoT")):
        if 'cot_reasoning' in example:
            continue  # Skip if already has reasoning
        
        try:
            options_text = "\n".join([
                f"{k}: {v}" for k, v in example.get('options', {}).items()
            ])
            
            prompt = cot_prompt.format(
                question=example.get('question', ''),
                options=options_text,
                answer=f"{example.get('answer_idx', '')} - {example.get('answer', '')}"
            )
            
            response = llm.invoke(prompt)
            reasoning = response.content if hasattr(response, 'content') else str(response)
            
            example['cot_reasoning'] = reasoning
            
        except Exception as e:
            print(f"Error generating CoT for example {i}: {e}")
            continue
        
        # Rate limiting
        if i > 0 and i % batch_size == 0:
            import time
            time.sleep(1)
    
    return examples


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training examples
    examples = load_jsonl(args.train_file, args.max_examples)
    
    if not examples:
        print("No examples loaded. Exiting.")
        return
    
    # Prepare examples
    print("Preparing examples...")
    prepared_examples = [prepare_example_with_metadata(ex) for ex in examples]
    
    # Optionally generate CoT reasoning
    if args.generate_cot:
        prepared_examples = generate_cot_reasoning(prepared_examples)
        
        # Save examples with CoT
        cot_filepath = output_dir / "examples_with_cot.jsonl"
        with open(cot_filepath, 'w', encoding='utf-8') as f:
            for ex in prepared_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"Saved examples with CoT to {cot_filepath}")
    
    # Initialize embedding service
    print(f"\nInitializing embedding model: {args.embedding_model}")
    embedding_service = EmbeddingService(model_name=args.embedding_model)
    
    # Build KNN index
    print("\nBuilding KNN index...")
    knn_retriever = KNNRetriever(
        embedding_service=embedding_service,
        index_path=str(output_dir / "train_index.pkl")
    )
    
    knn_retriever.build_index(
        examples=prepared_examples,
        text_key="question",
        include_options=args.include_options,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Save index
    knn_retriever.save_index()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Knowledge Base Build Complete!")
    print("=" * 60)
    print(f"Total examples indexed: {len(prepared_examples)}")
    print(f"Embedding dimension: {embedding_service.embedding_dim}")
    print(f"Model used: {embedding_service.model_name}")
    print(f"Index saved to: {output_dir / 'train_index.pkl'}")
    
    # Test retrieval
    print("\n" + "-" * 60)
    print("Testing retrieval with a sample query...")
    print("-" * 60)
    
    test_question = "A 45-year-old man presents with chest pain and shortness of breath."
    similar = knn_retriever.get_similar_examples(test_question, k=3)
    
    print(f"\nQuery: {test_question}")
    print(f"\nTop 3 similar examples:")
    for i, ex in enumerate(similar, 1):
        print(f"\n{i}. (similarity: {ex['similarity_score']:.4f})")
        print(f"   Question: {ex['question'][:100]}...")
        print(f"   Answer: {ex['answer_idx']} - {ex['answer']}")


if __name__ == "__main__":
    main()

