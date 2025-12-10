"""Quick start example for Multi-turn Chat."""

from workflows import create_chat_session
from utils.config import Config


def main():
    """Quick start demo."""
    print("="*60)
    print("Multi-turn Chat - Quick Start Demo")
    print("="*60)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease set up your .env file with:")
        print("  - GOOGLE_API_KEY (for Gemini)")
        print("  - TAVILY_API_KEY (for web search)")
        return
    
    # Create chat session
    print("\n📝 Creating chat session...")
    session = create_chat_session(
        session_id="quickstart",
        max_recent_turns=3,
        summarize_threshold=5,
        use_summarization=True,
        enable_medprompt=False  # Disable for faster demo
    )
    print("✓ Session created!")
    
    print("\n" + "="*60)
    print("Starting conversation...")
    print("="*60)
    
    # Turn 1
    print("\n👤 User: What is hypertension?")
    result1 = session.chat("What is hypertension?")
    print(f"🤖 Assistant: {result1['answer'][:300]}...")
    print(f"   [Workflow: {result1.get('workflow_used')}, Time: {result1.get('execution_time', 0):.2f}s]")
    
    # Turn 2 - Follow-up (uses context automatically)
    print("\n👤 User: What are the symptoms?")
    result2 = session.chat("What are the symptoms?")
    print(f"🤖 Assistant: {result2['answer'][:300]}...")
    print(f"   [Workflow: {result2.get('workflow_used')}, Time: {result2.get('execution_time', 0):.2f}s]")
    
    # Turn 3 - Another follow-up
    print("\n👤 User: How is it treated?")
    result3 = session.chat("How is it treated?")
    print(f"🤖 Assistant: {result3['answer'][:300]}...")
    print(f"   [Workflow: {result3.get('workflow_used')}, Time: {result3.get('execution_time', 0):.2f}s]")
    
    # Show session stats
    print("\n" + "="*60)
    print("Session Statistics")
    print("="*60)
    
    info = session.get_session_info()
    stats = info['memory_stats']
    
    print(f"✓ Total turns: {stats['total_turns']}")
    print(f"✓ Stored turns: {stats['stored_turns']}")
    print(f"✓ Has summary: {stats['has_summary']}")
    print(f"✓ Session duration: {info['duration']:.2f}s")
    
    print("\n" + "="*60)
    print("Quick Start Complete!")
    print("="*60)
    print("\n📖 Next steps:")
    print("  1. Run 'python example_multi_turn_chat.py' for more examples")
    print("  2. Read MULTI_TURN_GUIDE.md for detailed documentation")
    print("  3. Customize memory settings for your use case")
    print("\n💡 Key features:")
    print("  • Context is automatically maintained across turns")
    print("  • Old conversations are summarized to save tokens")
    print("  • Session can be exported and restored")
    print("  • Works with all SuperGraph features (routing, image, etc.)")


if __name__ == "__main__":
    main()


