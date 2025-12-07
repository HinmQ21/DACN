"""Main script to run medical QA system on single questions."""

import argparse
from workflows import MedicalQAWorkflow, create_workflow
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
    
    # Reflexion arguments
    parser.add_argument(
        '--reflexion',
        action='store_true',
        default=None,
        help='Enable self-correction (Reflexion)'
    )
    parser.add_argument(
        '--no-reflexion',
        action='store_true',
        help='Disable self-correction (Reflexion)'
    )
    
    args = parser.parse_args()
    
    # Determine reflexion setting
    enable_reflexion = None  # Use config default
    if args.reflexion:
        enable_reflexion = True
    elif args.no_reflexion:
        enable_reflexion = False
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file with required API keys.")
        return
    
    # Initialize workflow
    print("Initializing Medical QA Workflow...")
    workflow = create_workflow(enable_reflexion=enable_reflexion)
    
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
    
    # Display Reflexion info
    reflexion_info = result.get('reflexion', {})
    if reflexion_info.get('enabled'):
        print("\n" + "-"*60)
        print("REFLEXION (Self-Correction)")
        print("-"*60)
        print(f"Performed: {reflexion_info.get('performed', False)}")
        if reflexion_info.get('performed'):
            print(f"Iterations: {reflexion_info.get('iterations', 0)}")
            if reflexion_info.get('correction_applied'):
                print(f"Original answer: {reflexion_info.get('original_answer')}")
                print(f"Original confidence: {reflexion_info.get('original_confidence', 0):.2f}")
                print(f"Correction applied: Yes")
            else:
                print(f"Correction applied: No (answer kept)")
            print(f"Reason: {reflexion_info.get('reason', 'N/A')}")
    
    # Display Medprompt info
    medprompt_info = result.get('medprompt', {})
    if medprompt_info.get('enabled'):
        print("\n" + "-"*60)
        print("MEDPROMPT INFO")
        print("-"*60)
        print(f"Few-shot examples: {medprompt_info.get('few_shot_count', 0)}")
        print(f"CoT used: {medprompt_info.get('cot_used', False)}")
        print(f"Ensemble used: {medprompt_info.get('ensemble_used', False)}")
        if medprompt_info.get('ensemble_used'):
            print(f"Ensemble consistency: {medprompt_info.get('ensemble_consistency', 0):.2f}")
    
    if args.verbose:
        print("\n" + "-"*60)
        print("DETAILED ANALYSIS")
        print("-"*60)
        print(f"\nCoordinator Analysis:")
        print(json.dumps(result['coordinator_analysis'], indent=2, ensure_ascii=False))
        print(f"\nValidation Results:")
        print(json.dumps(result['validation'], indent=2, ensure_ascii=False))
        
        # Show full reflexion details in verbose mode
        if reflexion_info.get('performed') and reflexion_info.get('critique'):
            print(f"\nReflexion Critique:")
            print(reflexion_info.get('critique'))
    
    if result.get('error'):
        print(f"\nWarning: {result['error']}")


if __name__ == "__main__":
    main()

