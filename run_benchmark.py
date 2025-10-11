"""Script to run benchmark evaluation on medical QA datasets."""

import argparse
from workflows import MedicalQAWorkflow
from benchmarks import MedQAEvaluator, PubMedQAEvaluator
from utils.config import Config
import json
from datetime import datetime


def print_metrics(metrics: dict, dataset_name: str):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*60)
    print(f"{dataset_name} EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nAverage Confidence: {metrics['avg_confidence']:.4f}")
    print(f"Structured Parse Success Rate: {metrics.get('parsed_success_rate', 0):.2%} ({metrics.get('parsed_success_count', 0)}/{metrics['total_count']})")
    print(f"\nAverage Time per Question: {metrics['avg_time_per_question']:.2f}s")
    print(f"Total Time: {metrics['total_time']:.2f}s")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run Medical QA Benchmark Evaluation")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['medqa', 'pubmedqa', 'both'],
        default='medqa',
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        help='Dataset split (dev/test/train for MedQA, test/train for PubMedQA)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results'
    )
    
    args = parser.parse_args()
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file with required API keys.")
        return
    
    # Override max_samples from config if not specified
    if args.max_samples is None:
        args.max_samples = Config.MAX_SAMPLES
    
    save_results = args.save_results and not args.no_save
    
    # Initialize workflow
    print("Initializing Medical QA Workflow...")
    workflow = MedicalQAWorkflow()
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': Config.GEMINI_MODEL,
            'temperature': Config.TEMPERATURE,
            'max_samples': args.max_samples
        },
        'results': {}
    }
    
    # Run MedQA evaluation
    if args.dataset in ['medqa', 'both']:
        print(f"\n{'='*60}")
        print(f"Starting MedQA Evaluation on {args.split} split")
        print(f"{'='*60}\n")
        
        evaluator = MedQAEvaluator(workflow)
        try:
            result = evaluator.evaluate(
                split=args.split,
                max_samples=args.max_samples,
                save_results=save_results
            )
            print_metrics(result['metrics'], f"MedQA ({args.split})")
            results_summary['results']['medqa'] = result['metrics']
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error during MedQA evaluation: {e}")
    
    # Run PubMedQA evaluation
    if args.dataset in ['pubmedqa', 'both']:
        print(f"\n{'='*60}")
        print(f"Starting PubMedQA Evaluation on {args.split} split")
        print(f"{'='*60}\n")
        
        evaluator = PubMedQAEvaluator(workflow)
        try:
            # PubMedQA only has 'test' split in pqa_labeled
            pubmed_split = 'test' if args.split in ['test', 'dev'] else 'train'
            result = evaluator.evaluate(
                split=pubmed_split,
                max_samples=args.max_samples,
                save_results=save_results
            )
            print_metrics(result['metrics'], f"PubMedQA ({pubmed_split})")
            results_summary['results']['pubmedqa'] = result['metrics']
        except Exception as e:
            print(f"Error during PubMedQA evaluation: {e}")
    
    # Save summary
    if save_results and results_summary['results']:
        summary_path = f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nBenchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()

