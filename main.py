"""Main script to run medical QA system on single questions."""

import argparse
from workflows import MedicalQAWorkflow
from utils.config import Config
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Medical QA Multi-Agent System")
    parser.add_argument(
        '--question', 
        type=str, 
        required=True,
        help='Medical question to answer'
    )
    parser.add_argument(
        '--options',
        type=str,
        nargs='+',
        help='Multiple choice options (e.g., "A. Option1" "B. Option2")'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['multiple_choice', 'yes_no'],
        default='multiple_choice',
        help='Question type'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file with required API keys.")
        return
    
    # Initialize workflow
    print("Initializing Medical QA Workflow...")
    workflow = MedicalQAWorkflow()
    
    # Run question
    print(f"\nQuestion: {args.question}")
    
    # Convert options list to dict format (required for Medprompt)
    options_dict = None
    if args.options:
        options_dict = {}
        print("Options:")
        for opt in args.options:
            # Parse "A. Option text" format
            if '. ' in opt:
                key, value = opt.split('. ', 1)
                options_dict[key.strip()] = value.strip()
            elif '.' in opt:
                key, value = opt.split('.', 1)
                options_dict[key.strip()] = value.strip()
            else:
                # Fallback: use first character as key
                options_dict[opt[0]] = opt
            print(f"  {opt}")
    
    print("\nProcessing... (this may take a minute)")
    
    start_time = time.time()
    
    result = workflow.run(
        question=args.question,
        options=options_dict,
        question_type=args.type
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nExplanation: {result['explanation']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"Sources used: {result['sources_count']}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if args.verbose:
        print("\n" + "-"*60)
        print("DETAILED ANALYSIS")
        print("-"*60)
        print(f"\nCoordinator Analysis:")
        print(json.dumps(result['coordinator_analysis'], indent=2, ensure_ascii=False))
        print(f"\nValidation Results:")
        print(json.dumps(result['validation'], indent=2, ensure_ascii=False))
    
    if result.get('error'):
        print(f"\nWarning: {result['error']}")


if __name__ == "__main__":
    main()

