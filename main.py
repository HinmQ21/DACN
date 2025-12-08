"""Main script to run medical QA system on single questions."""

import argparse
from workflows import MedicalQAWorkflow, create_workflow, ImageQAWorkflow, create_image_workflow, detect_input_type
from utils.config import Config
import json
import time


def main():
    parser = argparse.ArgumentParser(description="Medical QA Multi-Agent System")
    parser.add_argument(
        '--question', 
        type=str, 
        required=False,
        help='Medical question to answer'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to medical image file or URL for image analysis/VQA'
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
    
    # Validate input: either question or image must be provided
    if not args.question and not args.image:
        parser.error("Either --question or --image must be provided")
    
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
    
    # Convert options list to dict format
    options_dict = None
    if args.options:
        options_dict = {}
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
    
    # Detect input type and route to appropriate workflow
    input_type = detect_input_type(question=args.question, image_input=args.image)
    
    if input_type == 'image':
        # Image workflow
        result = run_image_workflow(args, options_dict)
    else:
        # Text workflow
        result = run_text_workflow(args, options_dict, enable_reflexion)
    
    # Display results
    display_results(result, args)


def run_text_workflow(args, options_dict, enable_reflexion):
    """Run the text-based Medical QA workflow."""
    print("Initializing Medical QA Workflow...")
    workflow = create_workflow(enable_reflexion=enable_reflexion)
    
    print(f"\nQuestion: {args.question}")
    
    if options_dict:
        print("Options:")
        for key, value in options_dict.items():
            print(f"  {key}: {value}")
    
    print("\nProcessing... (this may take a minute)")
    
    start_time = time.time()
    
    result = workflow.run(
        question=args.question,
        options=options_dict,
        question_type=args.type
    )
    
    result['elapsed_time'] = time.time() - start_time
    result['workflow_type'] = 'text'
    
    return result


def run_image_workflow(args, options_dict):
    """Run the Image QA workflow."""
    print("Initializing Image QA Workflow...")
    workflow = create_image_workflow()
    
    print(f"\nImage: {args.image}")
    
    if args.question:
        print(f"Question: {args.question}")
        mode = "VQA (Visual Question Answering)"
    else:
        mode = "Image Analysis"
    
    print(f"Mode: {mode}")
    
    if options_dict:
        print("Options:")
        for key, value in options_dict.items():
            print(f"  {key}: {value}")
    
    print("\nProcessing... (this may take a minute)")
    
    start_time = time.time()
    
    result = workflow.run(
        image_input=args.image,
        question=args.question,
        options=options_dict
    )
    
    result['elapsed_time'] = time.time() - start_time
    result['workflow_type'] = 'image'
    
    return result


def display_results(result, args):
    """Display workflow results."""
    elapsed_time = result.get('elapsed_time', result.get('execution_time', 0))
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    workflow_type = result.get('workflow_type', 'text')
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nExplanation: {result['explanation']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    
    if workflow_type == 'text':
        print(f"Sources used: {result.get('sources_count', 0)}")
    else:
        # Image workflow specific info
        print(f"Mode: {result.get('mode', 'analysis')}")
        if result.get('image_type'):
            print(f"Image type: {result['image_type']}")
        if result.get('findings'):
            print(f"Findings: {len(result['findings'])} items")
    
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Display Reflexion info (text workflow only)
    if workflow_type == 'text':
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
    
    # Display Image Analysis info (image workflow)
    if workflow_type == 'image' and args.verbose:
        print("\n" + "-"*60)
        print("IMAGE ANALYSIS DETAILS")
        print("-"*60)
        image_analysis = result.get('image_analysis', {})
        if image_analysis.get('findings'):
            print("\nFindings:")
            for finding in image_analysis['findings']:
                print(f"  - {finding}")
        if image_analysis.get('raw_output'):
            print(f"\nRaw Analysis:\n{image_analysis['raw_output'][:1000]}...")
    
    if args.verbose and workflow_type == 'text':
        print("\n" + "-"*60)
        print("DETAILED ANALYSIS")
        print("-"*60)
        print(f"\nCoordinator Analysis:")
        print(json.dumps(result.get('coordinator_analysis', {}), indent=2, ensure_ascii=False))
        print(f"\nValidation Results:")
        print(json.dumps(result.get('validation', {}), indent=2, ensure_ascii=False))
        
        # Show full reflexion details in verbose mode
        reflexion_info = result.get('reflexion', {})
        if reflexion_info.get('performed') and reflexion_info.get('critique'):
            print(f"\nReflexion Critique:")
            print(reflexion_info.get('critique'))
    
    if result.get('error'):
        print(f"\nWarning: {result['error']}")


if __name__ == "__main__":
    main()

