"""Example usage of Multi-turn Chat with Memory Management."""

from workflows import create_chat_session
from utils.config import Config
import time


def example_basic_multi_turn():
    """Example: Basic multi-turn conversation."""
    print("="*60)
    print("Example 1: Basic Multi-turn Conversation")
    print("="*60)
    
    # Create chat session
    session = create_chat_session(
        session_id="demo_basic",
        max_recent_turns=3,
        summarize_threshold=5,
        use_summarization=True,
        enable_medprompt=False  # Disable for faster demo
    )
    
    # Turn 1: Ask about hypertension
    print("\n[Turn 1] User asks about hypertension...")
    result1 = session.chat("What is hypertension?")
    print(f"Assistant: {result1['answer'][:150]}...")
    print(f"Workflow: {result1.get('workflow_used')}, Time: {result1.get('execution_time', 0):.2f}s")
    
    time.sleep(1)
    
    # Turn 2: Follow-up question
    print("\n[Turn 2] User asks follow-up question...")
    result2 = session.chat("What are the symptoms?")
    print(f"Assistant: {result2['answer'][:150]}...")
    print(f"Workflow: {result2.get('workflow_used')}, Time: {result2.get('execution_time', 0):.2f}s")
    
    time.sleep(1)
    
    # Turn 3: Another follow-up
    print("\n[Turn 3] User asks about treatment...")
    result3 = session.chat("How is it treated?")
    print(f"Assistant: {result3['answer'][:150]}...")
    print(f"Workflow: {result3.get('workflow_used')}, Time: {result3.get('execution_time', 0):.2f}s")
    
    time.sleep(1)
    
    # Turn 4: Ask about complications
    print("\n[Turn 4] User asks about complications...")
    result4 = session.chat("What are the potential complications if left untreated?")
    print(f"Assistant: {result4['answer'][:150]}...")
    print(f"Workflow: {result4.get('workflow_used')}, Time: {result4.get('execution_time', 0):.2f}s")
    
    # Show conversation history
    print("\n" + "="*60)
    print("Conversation History")
    print("="*60)
    session.print_history()
    
    # Show session info
    session_info = session.get_session_info()
    print(f"\nSession Info:")
    print(f"  - Total turns: {session_info['memory_stats']['total_turns']}")
    print(f"  - Stored turns: {session_info['memory_stats']['stored_turns']}")
    print(f"  - Has summary: {session_info['memory_stats']['has_summary']}")


def example_topic_switch():
    """Example: Conversation with topic switch."""
    print("\n\n" + "="*60)
    print("Example 2: Multi-turn with Topic Switch")
    print("="*60)
    
    session = create_chat_session(
        session_id="demo_topic_switch",
        max_recent_turns=2,
        summarize_threshold=4,
        use_summarization=True,
        enable_medprompt=False
    )
    
    # Discuss diabetes
    print("\n[Topic 1: Diabetes]")
    session.chat("What is diabetes?")
    print("  ✓ Asked about diabetes")
    
    session.chat("What are the types of diabetes?")
    print("  ✓ Asked about types")
    
    time.sleep(1)
    
    # Switch to heart disease
    print("\n[Topic 2: Heart Disease]")
    result = session.chat("Now let's talk about heart disease. What causes it?")
    print(f"Assistant: {result['answer'][:150]}...")
    
    session.chat("How can it be prevented?")
    print("  ✓ Asked about prevention")
    
    time.sleep(1)
    
    # Reference both topics
    print("\n[Connecting Topics]")
    result = session.chat("Is there a connection between diabetes and heart disease?")
    print(f"Assistant: {result['answer'][:200]}...")
    
    # Show memory stats
    stats = session.get_session_info()['memory_stats']
    print(f"\nMemory Stats:")
    print(f"  - Total turns: {stats['total_turns']}")
    print(f"  - Stored turns: {stats['stored_turns']}")
    print(f"  - Summary created: {stats['has_summary']}")
    if stats['has_summary']:
        print(f"  - Summary length: {stats['summary_length']} chars")


def example_complex_multi_turn():
    """Example: Complex multi-turn conversation with medical reasoning."""
    print("\n\n" + "="*60)
    print("Example 3: Complex Multi-turn Medical Discussion")
    print("="*60)
    
    session = create_chat_session(
        session_id="demo_complex",
        max_recent_turns=3,
        summarize_threshold=5,
        use_summarization=True,
        enable_medprompt=True,  # Enable for complex questions
        enable_ensemble=False,  # Disable for speed
        enable_reflexion=False  # Disable for speed
    )
    
    # Start with a clinical scenario
    print("\n[Turn 1] Present clinical case...")
    result1 = session.chat(
        "A 55-year-old man presents with chest pain. What should be the initial workup?"
    )
    print(f"Assistant: {result1['answer'][:200]}...")
    print(f"Workflow: {result1.get('workflow_used')}")
    
    time.sleep(1)
    
    # Follow-up with more details
    print("\n[Turn 2] Add more clinical details...")
    result2 = session.chat(
        "The ECG shows ST elevation in leads II, III, and aVF. What does this indicate?"
    )
    print(f"Assistant: {result2['answer'][:200]}...")
    print(f"Workflow: {result2.get('workflow_used')}")
    
    time.sleep(1)
    
    # Ask about management
    print("\n[Turn 3] Ask about management...")
    result3 = session.chat(
        "What is the immediate management for this patient?"
    )
    print(f"Assistant: {result3['answer'][:200]}...")
    print(f"Workflow: {result3.get('workflow_used')}")
    
    # Show conversation history
    print("\n")
    session.print_history()


def example_long_conversation_with_summarization():
    """Example: Long conversation demonstrating summarization."""
    print("\n\n" + "="*60)
    print("Example 4: Long Conversation with Summarization")
    print("="*60)
    
    session = create_chat_session(
        session_id="demo_long",
        max_recent_turns=2,
        summarize_threshold=4,  # Trigger summarization after 4 turns
        use_summarization=True,
        enable_medprompt=False
    )
    
    questions = [
        "What is pneumonia?",
        "What causes it?",
        "What are the symptoms?",
        "How is it diagnosed?",
        "What are the treatment options?",
        "How long does recovery take?",
        "Can it be prevented?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Turn {i}] {question}")
        result = session.chat(question)
        
        stats = session.memory.get_stats()
        print(f"  ✓ Response received ({len(result['answer'])} chars)")
        print(f"  Memory: {stats['stored_turns']} turns stored, "
              f"Summary: {'Yes' if stats['has_summary'] else 'No'}")
        
        time.sleep(0.5)
    
    # Show final state
    print("\n" + "="*60)
    print("Final Memory State")
    print("="*60)
    
    stats = session.get_session_info()['memory_stats']
    print(f"Total turns: {stats['total_turns']}")
    print(f"Turns in memory: {stats['stored_turns']}")
    print(f"Has summary: {stats['has_summary']}")
    
    if stats['has_summary']:
        print(f"\nConversation Summary:")
        print("-" * 60)
        print(session.get_summary())
        print("-" * 60)
    
    print(f"\nRecent turns in memory:")
    for turn in session.memory.get_recent_turns():
        print(f"  Turn {turn.turn_id}: {turn.user_message[:50]}...")


def example_interactive_chat():
    """Example: Interactive chat demo (simulated)."""
    print("\n\n" + "="*60)
    print("Example 5: Interactive Chat Simulation")
    print("="*60)
    
    session = create_chat_session(
        session_id="demo_interactive",
        max_recent_turns=3,
        summarize_threshold=6,
        use_summarization=True,
        enable_medprompt=False
    )
    
    # Simulate user interactions
    conversation = [
        "Hi, I have a question about antibiotics",
        "What's the difference between amoxicillin and azithromycin?",
        "Which one is better for respiratory infections?",
        "Are there any side effects I should know about?",
        "Thank you! One more thing - can I take them with food?"
    ]
    
    print("\nStarting interactive chat simulation...\n")
    
    for i, user_input in enumerate(conversation, 1):
        print(f"{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")
        
        result = session.chat(user_input)
        
        print(f"\nAssistant: {result['answer']}\n")
        print(f"[Metadata]")
        print(f"  - Workflow: {result.get('workflow_used')}")
        print(f"  - Confidence: {result.get('confidence', 0):.2f}")
        print(f"  - Time: {result.get('execution_time', 0):.2f}s")
        print(f"  - Turn: {i}/{len(conversation)}")
        
        time.sleep(1)
    
    # Show final session info
    print(f"\n{'='*60}")
    print("Session Summary")
    print(f"{'='*60}")
    
    session_info = session.get_session_info()
    print(f"Session ID: {session_info['session_id']}")
    print(f"Duration: {session_info['duration']:.2f}s")
    print(f"Total turns: {session_info['memory_stats']['total_turns']}")
    
    if session_info['memory_stats']['has_summary']:
        print(f"\nConversation was summarized:")
        print(session.get_summary())


def main():
    """Run all examples."""
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file with required API keys.")
        return
    
    print("\n" + "="*60)
    print("Multi-turn Chat Examples - Memory Management Demo")
    print("="*60)
    print("\nThese examples demonstrate:")
    print("  1. Basic multi-turn conversations")
    print("  2. Topic switching with memory")
    print("  3. Complex medical discussions")
    print("  4. Automatic conversation summarization")
    print("  5. Interactive chat simulation")
    print("\n")
    
    # Run examples (uncomment the ones you want)
    example_basic_multi_turn()
    # example_topic_switch()
    # example_complex_multi_turn()
    # example_long_conversation_with_summarization()
    # example_interactive_chat()
    
    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-turn conversation tracking")
    print("  ✓ Automatic conversation summarization")
    print("  ✓ Context-aware responses")
    print("  ✓ Memory management with configurable thresholds")
    print("  ✓ Session persistence and export")


if __name__ == "__main__":
    main()


