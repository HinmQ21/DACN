"""Single LLM benchmark evaluation - đánh giá một LLM đơn không dùng multi-agent workflow."""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time
from datasets import load_dataset
import google.generativeai as genai
from utils.config import Config
from utils.metrics import calculate_metrics, extract_answer_from_text


class SingleLLMEvaluator:
    """Base class cho đánh giá single LLM."""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize single LLM evaluator.
        
        Args:
            model_name: Gemini model name (default: from config)
            temperature: Generation temperature (default: from config)
        """
        self.model_name = model_name or Config.GEMINI_MODEL
        self.temperature = temperature if temperature is not None else Config.TEMPERATURE
        
        # Configure Gemini API
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"Initialized Single LLM Evaluator with model: {self.model_name}, temperature: {self.temperature}")
    
    def generate_answer(self, question: str, options: List[str] = None) -> Dict[str, Any]:
        """
        Generate answer using single LLM call.
        
        Args:
            question: Question to answer
            options: Answer options (if any)
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build prompt
        if options:
            options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
            prompt = f"""You are a medical expert. Answer the following medical question.

Question: {question}

Options:
{options_text}

Please provide:
1. Your answer (just the option number or letter)
2. A brief explanation for your answer

Format your response as:
Answer: [your answer]
Explanation: [your explanation]"""
        else:
            prompt = f"""You are a medical expert. Answer the following medical question with yes, no, or maybe.

Question: {question}

Please provide:
1. Your answer (yes/no/maybe)
2. A brief explanation

Format your response as:
Answer: [your answer]
Explanation: [your explanation]"""
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                )
            )
            
            response_text = response.text
            
            # Parse response
            answer = ""
            explanation = ""
            
            lines = response_text.strip().split('\n')
            for i, line in enumerate(lines):
                if line.lower().startswith('answer:'):
                    answer = line.split(':', 1)[1].strip()
                elif line.lower().startswith('explanation:'):
                    explanation = '\n'.join(lines[i:]).split(':', 1)[1].strip()
                    break
            
            # If parsing failed, use the entire response as answer
            if not answer:
                answer = response_text.split('\n')[0]
                explanation = response_text
            
            return {
                'answer': answer,
                'explanation': explanation,
                'raw_response': response_text,
                'parsed_successfully': bool(answer and explanation),
                'confidence': 0.8,  # Default confidence for single LLM
                'sources_count': 0,
                'error': None
            }
            
        except Exception as e:
            return {
                'answer': '',
                'explanation': '',
                'raw_response': '',
                'parsed_successfully': False,
                'confidence': 0.0,
                'sources_count': 0,
                'error': str(e)
            }


class SingleLLMMedQAEvaluator(SingleLLMEvaluator):
    """Evaluator cho MedQA dataset với single LLM."""
    
    def load_dataset(self, split: str = "dev", max_samples: int = None) -> List[Dict]:
        """
        Load MedQA dataset.
        
        Args:
            split: 'dev', 'test', hoặc 'train'
            max_samples: Số lượng mẫu tối đa (None = tất cả)
            
        Returns:
            List of questions
        """
        data_path = Path(f"MedQA/{split}.jsonl")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        questions = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                questions.append(json.loads(line))
        
        return questions
    
    def safe_sleep(self, seconds, bar=None):
        """Sleep without affecting tqdm's time calculation."""
        if bar is not None:
            # Save time before sleep
            sleep_start = time.time()
            time.sleep(seconds)
            sleep_duration = time.time() - sleep_start
            
            # Adjust tqdm's internal timers to exclude sleep time
            if hasattr(bar, 'start_t'):
                bar.start_t += sleep_duration
            if hasattr(bar, 'last_print_t'):
                bar.last_print_t += sleep_duration
        else:
            time.sleep(seconds)
    
    def evaluate(
        self, 
        split: str = "dev", 
        max_samples: int = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá trên MedQA dataset.
        
        Args:
            split: Dataset split to evaluate on
            max_samples: Maximum number of samples
            save_results: Whether to save detailed results
            
        Returns:
            Evaluation metrics and results
        """
        print(f"Loading MedQA {split} dataset...")
        questions = self.load_dataset(split, max_samples)
        print(f"Loaded {len(questions)} questions")
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        total_time = 0
        
        with tqdm(questions, desc=f"Evaluating MedQA (Single LLM: {self.model_name})") as pbar:
            for idx, item in enumerate(pbar):
                try:
                    question_text = item['question']
                    options = item.get('options', {})
                    
                    # Format options
                    options_list = []
                    for key in sorted(options.keys()):
                        options_list.append(f"{key}. {options[key]}")
                    
                    # Get answer key
                    answer_key = item.get('answer_idx', item.get('answer', ''))
                    ground_truths.append(answer_key.upper())
                    
                    # Generate answer with single LLM
                    start_time = time.time()
                    result = self.generate_answer(
                        question=question_text,
                        options=options_list
                    )
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    
                    # Extract prediction
                    predicted_answer = extract_answer_from_text(
                        result['answer'], 
                        options=list(options.keys())
                    )
                    predictions.append(predicted_answer.upper())
                    
                    # Store detailed result
                    detailed_results.append({
                        'index': idx,
                        'question': question_text,
                        'options': options,
                        'ground_truth': answer_key,
                        'prediction': predicted_answer,
                        'is_correct': predicted_answer.upper() == answer_key.upper(),
                        'explanation': result.get('explanation', ''),
                        'confidence': result.get('confidence', 0.0),
                        'sources_count': result.get('sources_count', 0),
                        'parsed_successfully': result.get('parsed_successfully', False),
                        'time': elapsed_time,
                        'error': result.get('error')
                    })
                    
                except Exception as e:
                    print(f"\nError processing question {idx}: {e}")
                    predictions.append("")
                    detailed_results.append({
                        'index': idx,
                        'error': str(e),
                        'is_correct': False
                    })

                # Sleep 1 second to avoid rate limiting
                self.safe_sleep(5, pbar)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        metrics['avg_time_per_question'] = total_time / len(questions) if questions else 0
        metrics['total_time'] = total_time
        
        # Calculate additional stats
        correct_count = sum(1 for r in detailed_results if r.get('is_correct', False))
        metrics['correct_count'] = correct_count
        metrics['total_count'] = len(questions)
        
        avg_confidence = sum(r.get('confidence', 0) for r in detailed_results) / len(detailed_results) if detailed_results else 0
        metrics['avg_confidence'] = avg_confidence
        
        # Track structured parsing success rate
        parsed_success_count = sum(1 for r in detailed_results if r.get('parsed_successfully', False))
        metrics['parsed_success_rate'] = parsed_success_count / len(detailed_results) if detailed_results else 0
        metrics['parsed_success_count'] = parsed_success_count
        
        # Save results
        if save_results:
            results_path = Path(f"results_single_llm_medqa_{split}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'metrics': metrics,
                    'detailed_results': detailed_results
                }, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {results_path}")
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results
        }


class SingleLLMPubMedQAEvaluator(SingleLLMEvaluator):
    """Evaluator cho PubMedQA dataset với single LLM."""
    
    def load_dataset(self, split: str = "test", max_samples: int = None) -> List[Dict]:
        """
        Load PubMedQA dataset from HuggingFace.
        
        Args:
            split: 'train' hoặc 'test'
            max_samples: Số lượng mẫu tối đa
            
        Returns:
            List of questions
        """
        print(f"Loading PubMedQA from HuggingFace...")
        
        # Load from HuggingFace datasets
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
        
        questions = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            questions.append({
                'question': item['question'],
                'context': item.get('context', {}),
                'long_answer': item.get('long_answer', ''),
                'final_decision': item.get('final_decision', ''),
                'pubid': item.get('pubid', '')
            })
        
        return questions
    
    def evaluate(
        self,
        split: str = "test",
        max_samples: int = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá trên PubMedQA dataset.
        
        Args:
            split: Dataset split
            max_samples: Maximum samples
            save_results: Whether to save results
            
        Returns:
            Evaluation metrics and results
        """
        print(f"Loading PubMedQA {split} dataset...")
        questions = self.load_dataset(split, max_samples)
        print(f"Loaded {len(questions)} questions")
        
        predictions = []
        ground_truths = []
        detailed_results = []
        
        total_time = 0
        
        for idx, item in enumerate(tqdm(questions, desc=f"Evaluating PubMedQA (Single LLM: {self.model_name})")):
            try:
                question_text = item['question']
                
                # Add context if available
                context = item.get('context', {})
                if context:
                    contexts_text = context.get('contexts', [])
                    if contexts_text:
                        question_with_context = f"{question_text}\n\nContext: {' '.join(contexts_text[:2])}"
                    else:
                        question_with_context = question_text
                else:
                    question_with_context = question_text
                
                # Ground truth (yes/no/maybe)
                ground_truth = item.get('final_decision', '').lower()
                ground_truths.append(ground_truth)
                
                # Generate answer with single LLM
                start_time = time.time()
                result = self.generate_answer(
                    question=question_with_context,
                    options=['yes', 'no', 'maybe']
                )
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Extract prediction
                predicted_answer = extract_answer_from_text(
                    result['answer'],
                    options=['yes', 'no', 'maybe']
                )
                predictions.append(predicted_answer.lower())
                
                # Store detailed result
                detailed_results.append({
                    'index': idx,
                    'pubid': item.get('pubid', ''),
                    'question': question_text,
                    'ground_truth': ground_truth,
                    'prediction': predicted_answer.lower(),
                    'is_correct': predicted_answer.lower() == ground_truth,
                    'explanation': result.get('explanation', ''),
                    'confidence': result.get('confidence', 0.0),
                    'sources_count': result.get('sources_count', 0),
                    'parsed_successfully': result.get('parsed_successfully', False),
                    'time': elapsed_time,
                    'error': result.get('error')
                })
                
                # Sleep to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                predictions.append("")
                detailed_results.append({
                    'index': idx,
                    'error': str(e),
                    'is_correct': False
                })
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        metrics['avg_time_per_question'] = total_time / len(questions) if questions else 0
        metrics['total_time'] = total_time
        
        # Additional stats
        correct_count = sum(1 for r in detailed_results if r.get('is_correct', False))
        metrics['correct_count'] = correct_count
        metrics['total_count'] = len(questions)
        
        avg_confidence = sum(r.get('confidence', 0) for r in detailed_results) / len(detailed_results) if detailed_results else 0
        metrics['avg_confidence'] = avg_confidence
        
        # Track structured parsing success rate
        parsed_success_count = sum(1 for r in detailed_results if r.get('parsed_successfully', False))
        metrics['parsed_success_rate'] = parsed_success_count / len(detailed_results) if detailed_results else 0
        metrics['parsed_success_count'] = parsed_success_count
        
        # Save results
        if save_results:
            results_path = Path(f"results_single_llm_pubmedqa_{split}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'metrics': metrics,
                    'detailed_results': detailed_results
                }, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {results_path}")
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results
        }

