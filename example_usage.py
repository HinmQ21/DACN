"""Example usage of the Medical QA Multi-Agent System with Medprompt."""

from workflows import MedicalQAWorkflow, create_workflow, ImageQAWorkflow, create_image_workflow, detect_input_type
from utils.config import Config


def example_multiple_choice():
    """Example: Multiple choice question with Medprompt features."""
    print("="*60)
    print("Example 1: Multiple Choice Question (with Medprompt)")
    print("="*60)
    
    # Create workflow with Medprompt features
    workflow = MedicalQAWorkflow()
    
    question = """A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. 
    Physical examination reveals dry skin and delayed reflexes. 
    What is the most likely diagnosis?"""
    
    # Use dict format for options (required for choice shuffling)
    options = {
        "A": "Hyperthyroidism",
        "B": "Hypothyroidism", 
        "C": "Cushing's syndrome",
        "D": "Addison's disease"
    }
    
    result = workflow.run(
        question=question,
        options=options,
        question_type="multiple_choice"
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nOptions:")
    for k, v in options.items():
        print(f"  {k}: {v}")
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {result['sources_count']}")
    
    # Show Medprompt-specific info
    medprompt_info = result.get('medprompt', {})
    print(f"\n--- Medprompt Info ---")
    print(f"Few-shot examples used: {medprompt_info.get('few_shot_count', 0)}")
    print(f"CoT reasoning: {medprompt_info.get('cot_used', False)}")
    print(f"Ensemble used: {medprompt_info.get('ensemble_used', False)}")
    
    if medprompt_info.get('ensemble_used'):
        print(f"Ensemble consistency: {medprompt_info.get('ensemble_consistency', 0):.2f}")
        print(f"All predictions: {medprompt_info.get('all_predictions', [])}")
        print(f"Vote distribution: {medprompt_info.get('vote_distribution', {})}")
    
    # Show Reflexion info
    reflexion_info = result.get('reflexion', {})
    print(f"\n--- Reflexion (Self-Correction) Info ---")
    print(f"Reflexion enabled: {reflexion_info.get('enabled', False)}")
    print(f"Reflexion performed: {reflexion_info.get('performed', False)}")
    if reflexion_info.get('performed'):
        print(f"Iterations: {reflexion_info.get('iterations', 0)}")
        if reflexion_info.get('correction_applied'):
            print(f"Original answer: {reflexion_info.get('original_answer')}")
            print(f"Original confidence: {reflexion_info.get('original_confidence', 0):.2f}")
            print(f"Correction applied: Yes")
        print(f"Reason: {reflexion_info.get('reason', '')}")
        if reflexion_info.get('critique'):
            print(f"Critique: {reflexion_info.get('critique')}")
    
    # Debug: Show actual error if any
    if result.get('error'):
        print(f"\n[DEBUG] Error: {result['error']}")


def example_yes_no():
    """Example: Yes/No question."""
    print("\n\n" + "="*60)
    print("Example 2: Yes/No Question")
    print("="*60)
    
    workflow = MedicalQAWorkflow()
    
    question = "Does metformin reduce the risk of cardiovascular events in patients with type 2 diabetes?"
    
    result = workflow.run(
        question=question,
        question_type="yes_no"
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']:.2f}")


def example_clinical_reasoning():
    """Example: Clinical reasoning question with Medprompt."""
    print("\n\n" + "="*60)
    print("Example 3: Clinical Reasoning (with Medprompt)")
    print("="*60)
    
    workflow = MedicalQAWorkflow()
    
    question = """A 60-year-old man with a history of hypertension presents with sudden onset 
    severe chest pain radiating to the back. Blood pressure is 180/110 mmHg in the right arm 
    and 140/90 mmHg in the left arm. What is the most appropriate next step?"""
    
    options = {
        "A": "Immediate thrombolysis",
        "B": "CT angiography of the chest",
        "C": "Echocardiography",
        "D": "Cardiac catheterization"
    }
    
    result = workflow.run(
        question=question,
        options=options,
        question_type="multiple_choice"
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nOptions:")
    for k, v in options.items():
        print(f"  {k}: {v}")
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Show Medprompt info
    medprompt_info = result.get('medprompt', {})
    if medprompt_info.get('ensemble_used'):
        print(f"\n--- Ensemble Results ---")
        print(f"Consistency: {medprompt_info.get('ensemble_consistency', 0):.2f}")
        print(f"Predictions: {medprompt_info.get('all_predictions', [])}")
    
    # Show Reflexion info
    reflexion_info = result.get('reflexion', {})
    if reflexion_info.get('performed'):
        print(f"\n--- Reflexion Results ---")
        print(f"Iterations: {reflexion_info.get('iterations', 0)}")
        print(f"Correction applied: {reflexion_info.get('correction_applied', False)}")


def example_without_medprompt():
    """Example: Run without Medprompt features for comparison."""
    print("\n\n" + "="*60)
    print("Example 4: Without Medprompt (for comparison)")
    print("="*60)
    
    # Create workflow without Medprompt
    workflow = create_workflow(enable_medprompt=False)
    
    question = """A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. 
    Physical examination reveals dry skin and delayed reflexes. 
    What is the most likely diagnosis?"""
    
    options = {
        "A": "Hyperthyroidism",
        "B": "Hypothyroidism", 
        "C": "Cushing's syndrome",
        "D": "Addison's disease"
    }
    
    result = workflow.run(
        question=question,
        options=options,
        question_type="multiple_choice"
    )
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    medprompt_info = result.get('medprompt', {})
    print(f"\nMedprompt enabled: {medprompt_info.get('enabled', False)}")


def example_image_analysis():
    """Example: Medical image analysis (X-ray, CT, etc.)."""
    print("\n\n" + "="*60)
    print("Example 5: Medical Image Analysis")
    print("="*60)
    
    # Create image workflow
    workflow = create_image_workflow()
    
    # Example with a sample medical image URL
    # You can replace with local file path: "./images/chest_xray.jpg"
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    
    print(f"\nAnalyzing image: {image_url}")
    print("(This is a sample chest X-ray from Wikipedia)")
    
    result = workflow.analyze(image_input=image_url)
    
    print(f"\n--- Results ---")
    print(f"Success: {result.get('image_analysis', {}).get('success', False)}")
    print(f"Image Type: {result.get('image_type', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result.get('findings'):
        print(f"\nFindings:")
        for finding in result['findings']:
            print(f"  - {finding}")
    
    print(f"\nInterpretation: {result['answer']}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


def example_image_vqa():
    """Example: Visual Question Answering on medical image."""
    print("\n\n" + "="*60)
    print("Example 6: Medical Image VQA (Visual Question Answering)")
    print("="*60)
    
    # Create image workflow
    workflow = create_image_workflow()
    
    # Example with a sample medical image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"

    # Question about the image
    question = "Is there any sign of cardiomegaly (enlarged heart) in this chest X-ray?"
    
    print(f"\nImage: {image_url}")
    print(f"Question: {question}")
    
    result = workflow.ask(
        image_input=image_url,
        question=question
    )
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nExplanation: {result['explanation'][:500]}..." if len(result.get('explanation', '')) > 500 else f"\nExplanation: {result.get('explanation', 'N/A')}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


def example_image_vqa_multiple_choice():
    """Example: VQA with multiple choice options."""
    print("\n\n" + "="*60)
    print("Example 7: Medical Image VQA with Multiple Choice")
    print("="*60)
    
    # Create image workflow
    workflow = create_image_workflow()
    
    # Example with a sample medical image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"

    # Multiple choice question
    question = "What type of medical imaging is shown?"
    options = {
        "A": "MRI scan",
        "B": "CT scan",
        "C": "Chest X-ray",
        "D": "Ultrasound"
    }
    
    print(f"\nImage: {image_url}")
    print(f"Question: {question}")
    print("Options:")
    for k, v in options.items():
        print(f"  {k}: {v}")
    
    result = workflow.ask(
        image_input=image_url,
        question=question,
        options=options
    )
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result.get('error'):
        print(f"\nError: {result['error']}")


def example_auto_detect_workflow():
    """Example: Automatically detect and route to appropriate workflow."""
    print("\n\n" + "="*60)
    print("Example 8: Auto-detect Input Type")
    print("="*60)
    
    # Test 1: Text only
    input_type = detect_input_type(question="What is diabetes?", image_input=None)
    print(f"Text only -> Detected: {input_type}")
    
    # Test 2: Image only
    input_type = detect_input_type(question=None, image_input="chest_xray.jpg")
    print(f"Image only -> Detected: {input_type}")
    
    # Test 3: Image + Question (VQA)
    input_type = detect_input_type(question="What is shown?", image_input="chest_xray.jpg")
    print(f"Image + Question -> Detected: {input_type}")


def example_show_config():
    """Show current Medprompt configuration."""
    print("="*60)
    print("Current Medprompt Configuration")
    print("="*60)
    
    Config.print_config()


def main():
    """Run all examples."""
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file with required API keys.")
        print("\nCopy .env.example to .env and add your API keys:")
        print("  - GOOGLE_API_KEY (Gemini)")
        print("  - TAVILY_API_KEY (Tavily)")
        return
    
    print("\nMedical QA Multi-Agent System - Examples with Medprompt\n")
    
    # Show configuration
    example_show_config()
    
    # Run examples
    print("\n")
    # example_multiple_choice()
    
    # # Uncomment to run more examples
    # example_yes_no()
    # example_clinical_reasoning()
    # example_without_medprompt()
    
    # # Image-related examples (uncomment to run)
    # example_image_analysis()
    # example_image_vqa()
    example_image_vqa_multiple_choice()
    # example_auto_detect_workflow()
    
    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
