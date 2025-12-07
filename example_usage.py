"""Example usage of the Medical QA Multi-Agent System with Medprompt."""

from workflows import MedicalQAWorkflow, create_workflow
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
    example_multiple_choice()
    
    # # Uncomment to run more examples
    # example_yes_no()
    # example_clinical_reasoning()
    # example_without_medprompt()
    
    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
