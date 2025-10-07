"""Example usage of the Medical QA Multi-Agent System."""

from workflows import MedicalQAWorkflow
from utils.config import Config


def example_multiple_choice():
    """Example: Multiple choice question."""
    print("="*60)
    print("Example 1: Multiple Choice Question")
    print("="*60)
    
    workflow = MedicalQAWorkflow()
    
    question = """A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. 
    Physical examination reveals dry skin and delayed reflexes. 
    What is the most likely diagnosis?"""
    
    options = [
        "A. Hyperthyroidism",
        "B. Hypothyroidism", 
        "C. Cushing's syndrome",
        "D. Addison's disease"
    ]
    
    result = workflow.run(
        question=question,
        options=options,
        question_type="multiple_choice"
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nOptions: {', '.join(options)}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {result['sources_count']}")


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
    """Example: Clinical reasoning question."""
    print("\n\n" + "="*60)
    print("Example 3: Clinical Reasoning")
    print("="*60)
    
    workflow = MedicalQAWorkflow()
    
    question = """A 60-year-old man with a history of hypertension presents with sudden onset 
    severe chest pain radiating to the back. Blood pressure is 180/110 mmHg in the right arm 
    and 140/90 mmHg in the left arm. What is the most appropriate next step?"""
    
    options = [
        "A. Immediate thrombolysis",
        "B. CT angiography of the chest",
        "C. Echocardiography",
        "D. Cardiac catheterization"
    ]
    
    result = workflow.run(
        question=question,
        options=options,
        question_type="multiple_choice"
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nOptions: {', '.join(options)}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']:.2f}")


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
    
    print("\nMedical QA Multi-Agent System - Examples\n")
    
    # Run examples
    example_multiple_choice()
    example_yes_no()
    example_clinical_reasoning()
    
    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()

