"""MedQA benchmark evaluation."""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time
from workflows import MedicalQAWorkflow
from utils.metrics import calculate_metrics, extract_answer_from_text


class MedQAEvaluator:
    """Evaluator cho MedQA dataset."""
    
    def __init__(self, workflow: MedicalQAWorkflow):
        self.workflow = workflow
    
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
        
        with tqdm(questions, desc="Evaluating MedQA") as pbar:
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
                    
                    # Run workflow
                    start_time = time.time()
                    result = self.workflow.run(
                        question=question_text,
                        options=options_list,
                        question_type="multiple_choice"
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
                    print(f"Error processing question {idx}: {e}")
                    predictions.append("")
                    detailed_results.append({
                        'index': idx,
                        'error': str(e),
                        'is_correct': False
                    })

                # sleep 15s
                self.safe_sleep(15, pbar)
        
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
            results_path = Path(f"results_medqa_{split}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': metrics,
                    'detailed_results': detailed_results
                }, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {results_path}")
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results
        }

