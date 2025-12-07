"""PubMedQA benchmark evaluation."""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import time
from datasets import load_dataset
from workflows import MedicalQAWorkflow
from utils.metrics import calculate_metrics, extract_answer_from_text


class PubMedQAEvaluator:
    """Evaluator cho PubMedQA dataset."""
    
    def __init__(self, workflow: MedicalQAWorkflow):
        self.workflow = workflow
    
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
        
        for idx, item in enumerate(tqdm(questions, desc="Evaluating PubMedQA")):
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
                
                # Run workflow - pass options as dict for consistency
                start_time = time.time()
                result = self.workflow.run(
                    question=question_with_context,
                    options={'yes': 'Yes', 'no': 'No', 'maybe': 'Maybe'},
                    question_type="yes_no"
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
            results_path = Path(f"results_pubmedqa_{split}.json")
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

