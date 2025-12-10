"""Example usage of the Super Graph with intelligent routing."""

from workflows import create_super_graph
from utils.config import Config


def example_simple_question():
    """Example: Simple question answered directly by coordinator."""
    print("="*60)
    print("Example 1: Simple Question (Direct Answer)")
    print("="*60)
    
    # Create super graph
    workflow = create_super_graph()
    
    question = "What is hypertension?"
    
    print(f"\nQuestion: {question}")
    print("\nProcessing...")
    
    result = workflow.run(question=question)
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    # Show routing decision
    routing = result.get('routing_decision', {})
    print(f"\n--- Routing Decision ---")
    print(f"Complexity: {routing.get('complexity', 'N/A')}")
    print(f"Can answer directly: {routing.get('can_answer_directly', False)}")
    print(f"Reasoning: {routing.get('reasoning', 'N/A')}")


def example_complex_question():
    """Example: Complex question routed to Medical QA subgraph."""
    print("\n\n" + "="*60)
    print("Example 2: Complex Medical Question (Medical QA Subgraph)")
    print("="*60)
    
    workflow = create_super_graph()
    
    question = """A 45-year-old man presents with chest pain, dyspnea, and diaphoresis. 
    His ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?"""
    
    options = {
        "A": "Anterior wall myocardial infarction",
        "B": "Inferior wall myocardial infarction", 
        "C": "Pulmonary embolism",
        "D": "Aortic dissection"
    }
    
    print(f"\nQuestion: {question}")
    print("\nOptions:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    print("\nProcessing...")
    
    result = workflow.run(question=question, options=options)
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    # Show routing decision
    routing = result.get('routing_decision', {})
    print(f"\n--- Routing Decision ---")
    print(f"Complexity: {routing.get('complexity', 'N/A')}")
    print(f"Requires medical research: {routing.get('requires_medical_research', False)}")
    print(f"Reasoning: {routing.get('reasoning', 'N/A')}")
    
    # Show Medprompt info if medical_qa was used
    medprompt_info = result.get('medprompt', {})
    if medprompt_info.get('enabled'):
        print(f"\n--- Medprompt Info ---")
        print(f"Few-shot examples: {medprompt_info.get('few_shot_count', 0)}")
        print(f"CoT used: {medprompt_info.get('cot_used', False)}")
        print(f"Ensemble used: {medprompt_info.get('ensemble_used', False)}")
        if medprompt_info.get('ensemble_used'):
            print(f"Ensemble consistency: {medprompt_info.get('ensemble_consistency', 0):.2f}")


def example_moderate_question():
    """Example: Moderate complexity question."""
    print("\n\n" + "="*60)
    print("Example 3: Moderate Complexity Question")
    print("="*60)
    
    workflow = create_super_graph()
    
    question = """What is the mechanism of action of ACE inhibitors and what are their main 
    indications in cardiovascular disease?"""
    
    print(f"\nQuestion: {question}")
    print("\nProcessing...")
    
    result = workflow.run(question=question)
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    routing = result.get('routing_decision', {})
    print(f"\n--- Routing Decision ---")
    print(f"Complexity: {routing.get('complexity', 'N/A')}")
    print(f"Reasoning: {routing.get('reasoning', 'N/A')}")


def example_multiple_simple_questions():
    """Example: Test multiple simple questions in sequence."""
    print("\n\n" + "="*60)
    print("Example 4: Multiple Simple Questions")
    print("="*60)
    
    workflow = create_super_graph()
    
    questions = [
        "What is diabetes?",
        "What are the symptoms of pneumonia?",
        "What does BMI stand for?",
        "What is the normal human body temperature?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = workflow.run(question=question)
        print(f"   Workflow: {result.get('workflow_used', 'N/A')}")
        print(f"   Answer: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"   Answer: {result['answer']}")
        print(f"   Time: {result.get('execution_time', 0):.2f}s")


def example_image_analysis():
    """Example: Image analysis routed to Image QA subgraph."""
    print("\n\n" + "="*60)
    print("Example 5: Image Analysis (Image QA Subgraph)")
    print("="*60)
    
    workflow = create_super_graph()
    
    # Example with a sample medical image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    
    print(f"\nImage: {image_url}")
    print("(Sample chest X-ray from Wikipedia)")
    print("\nProcessing...")
    
    result = workflow.run(image_input=image_url)
    
    print(f"\n--- Results ---")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Mode: {result.get('mode', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    if result.get('findings'):
        print(f"\nFindings:")
        for finding in result['findings'][:5]:  # Show first 5
            print(f"  - {finding}")
    
    print(f"\nInterpretation: {result['answer'][:200]}..." if len(result['answer']) > 200 else f"\nInterpretation: {result['answer']}")
    
    # Show routing decision
    routing = result.get('routing_decision', {})
    print(f"\n--- Routing Decision ---")
    print(f"Input type: {routing.get('input_type', 'N/A')}")
    print(f"Requires image analysis: {routing.get('requires_image_analysis', False)}")


def example_image_vqa():
    """Example: Visual question answering with image."""
    print("\n\n" + "="*60)
    print("Example 6: Image + Question (VQA)")
    print("="*60)
    
    workflow = create_super_graph()
    
    image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    question = "Is there any sign of cardiomegaly in this chest X-ray?"
    
    print(f"\nImage: {image_url}")
    print(f"Question: {question}")
    print("\nProcessing...")
    
    result = workflow.run(
        image_input=image_url,
        question=question
    )
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")


def example_mixed_complexity():
    """Example: Test routing with questions of different complexity."""
    print("\n\n" + "="*60)
    print("Example 7: Mixed Complexity Questions")
    print("="*60)
    
    workflow = create_super_graph()
    
    test_cases = [
        ("Simple", "What is aspirin?"),
        ("Moderate", "What are the contraindications for aspirin use?"),
        ("Complex", "A 65-year-old patient with history of GI bleeding is on warfarin for atrial fibrillation. Should aspirin be added for cardiovascular prophylaxis?")
    ]
    
    for complexity_label, question in test_cases:
        print(f"\n[{complexity_label}] Question: {question}")
        result = workflow.run(question=question)
        
        routing = result.get('routing_decision', {})
        print(f"  Detected complexity: {routing.get('complexity', 'N/A')}")
        print(f"  Workflow used: {result.get('workflow_used', 'N/A')}")
        print(f"  Time: {result.get('execution_time', 0):.2f}s")


def example_custom_configuration():
    """Example: Create super graph with custom configuration."""
    print("\n\n" + "="*60)
    print("Example 8: Custom Configuration")
    print("="*60)
    
    # Create with custom settings
    workflow = create_super_graph(
        enable_medprompt=True,
        enable_few_shot=True,
        enable_cot=True,
        enable_ensemble=False,  # Disable ensemble for faster processing
        enable_reflexion=False  # Disable reflexion for faster processing
    )
    
    question = """A 30-year-old woman presents with polyuria, polydipsia, and weight loss. 
    Random blood glucose is 280 mg/dL. What is the most likely diagnosis?"""
    
    options = {
        "A": "Type 1 diabetes mellitus",
        "B": "Type 2 diabetes mellitus",
        "C": "Diabetes insipidus",
        "D": "Cushing's syndrome"
    }
    
    print(f"\nQuestion: {question}")
    print("\nConfiguration: Medprompt enabled, but ensemble and reflexion disabled")
    print("\nProcessing...")
    
    result = workflow.run(question=question, options=options)
    
    print(f"\n--- Results ---")
    print(f"Answer: {result['answer']}")
    print(f"Workflow used: {result.get('workflow_used', 'N/A')}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    medprompt_info = result.get('medprompt', {})
    print(f"\n--- Medprompt Status ---")
    print(f"Enabled: {medprompt_info.get('enabled', False)}")
    print(f"Few-shot examples: {medprompt_info.get('few_shot_count', 0)}")
    print(f"Ensemble used: {medprompt_info.get('ensemble_used', False)}")
    
    reflexion_info = result.get('reflexion', {})
    print(f"\n--- Reflexion Status ---")
    print(f"Enabled: {reflexion_info.get('enabled', False)}")


def example_show_config():
    """Show current configuration."""
    print("="*60)
    print("Current System Configuration")
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
        print("\nRequired keys:")
        print("  - GOOGLE_API_KEY (for Gemini)")
        print("  - TAVILY_API_KEY (for web search)")
        return
    
    print("\nSuper Graph Examples - Intelligent Routing Demo\n")
    
    # Show configuration
    example_show_config()
    
    # Run examples
    print("\n")
    
    # Uncomment the examples you want to run
    example_simple_question()
    # example_complex_question()
    # example_moderate_question()
    # example_multiple_simple_questions()
    # example_mixed_complexity()
    # example_custom_configuration()
    
    # Image examples (uncomment to run - requires image model)
    example_image_analysis()
    example_image_vqa()
    
    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nTips:")
    print("  - Simple questions are answered directly (fastest)")
    print("  - Complex questions use Medical QA subgraph (most accurate)")
    print("  - Image inputs use Image QA subgraph (specialized)")
    print("  - Check routing_decision in results to see how questions are classified")


if __name__ == "__main__":
    main()

