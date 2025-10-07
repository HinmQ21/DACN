"""Test script to verify per-agent configuration."""

from utils.config import Config


def test_config():
    """Test and display configuration for all agents."""
    print("="*60)
    print("Agent Configuration Test")
    print("="*60)
    
    agents = [
        ('default', None),
        ('coordinator', 'coordinator'),
        ('reasoning', 'reasoning'),
        ('validator', 'validator'),
        ('answer_generator', 'answer_generator'),
        ('web_search', 'web_search')
    ]
    
    print(f"\n{'Agent':<20} {'Model':<30} {'Temperature':<12}")
    print("-"*60)
    
    for name, agent_name in agents:
        config = Config.get_llm_config(agent_name)
        print(f"{name:<20} {config['model']:<30} {config['temperature']:<12.2f}")
    
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Default Model: {Config.GEMINI_MODEL}")
    print(f"Default Temperature: {Config.TEMPERATURE}")
    
    # Check for overrides
    overrides = []
    if Config.COORDINATOR_MODEL:
        overrides.append(f"  - Coordinator: {Config.COORDINATOR_MODEL}")
    if Config.REASONING_MODEL:
        overrides.append(f"  - Reasoning: {Config.REASONING_MODEL}")
    if Config.VALIDATOR_MODEL:
        overrides.append(f"  - Validator: {Config.VALIDATOR_MODEL}")
    if Config.ANSWER_GENERATOR_MODEL:
        overrides.append(f"  - Answer Generator: {Config.ANSWER_GENERATOR_MODEL}")
    if Config.WEB_SEARCH_MODEL:
        overrides.append(f"  - Web Search: {Config.WEB_SEARCH_MODEL}")
    
    if overrides:
        print("\nModel Overrides:")
        for override in overrides:
            print(override)
    else:
        print("\nNo model overrides (all agents use default model)")
    
    # Temperature overrides
    temp_overrides = []
    if Config.COORDINATOR_TEMPERATURE > 0:
        temp_overrides.append(f"  - Coordinator: {Config.COORDINATOR_TEMPERATURE}")
    if Config.REASONING_TEMPERATURE > 0:
        temp_overrides.append(f"  - Reasoning: {Config.REASONING_TEMPERATURE}")
    if Config.VALIDATOR_TEMPERATURE > 0:
        temp_overrides.append(f"  - Validator: {Config.VALIDATOR_TEMPERATURE}")
    if Config.ANSWER_GENERATOR_TEMPERATURE > 0:
        temp_overrides.append(f"  - Answer Generator: {Config.ANSWER_GENERATOR_TEMPERATURE}")
    if Config.WEB_SEARCH_TEMPERATURE > 0:
        temp_overrides.append(f"  - Web Search: {Config.WEB_SEARCH_TEMPERATURE}")
    
    if temp_overrides:
        print("\nTemperature Overrides:")
        for override in temp_overrides:
            print(override)
    else:
        print("\nNo temperature overrides (all agents use default temperature)")
    
    print("\n" + "="*60)


def test_agent_initialization():
    """Test that agents can be initialized with their configs."""
    print("\n" + "="*60)
    print("Agent Initialization Test")
    print("="*60)
    
    try:
        from agents import (
            CoordinatorAgent,
            ReasoningAgent,
            ValidatorAgent,
            AnswerGeneratorAgent,
            WebSearchAgent
        )
        
        agents = {
            'CoordinatorAgent': CoordinatorAgent,
            'ReasoningAgent': ReasoningAgent,
            'ValidatorAgent': ValidatorAgent,
            'AnswerGeneratorAgent': AnswerGeneratorAgent,
            'WebSearchAgent': WebSearchAgent
        }
        
        for name, AgentClass in agents.items():
            try:
                agent = AgentClass()
                print(f"✓ {name:<25} initialized successfully")
            except Exception as e:
                print(f"✗ {name:<25} failed: {e}")
        
        print("\n" + "="*60)
        
    except ImportError as e:
        print(f"Error importing agents: {e}")
        print("Make sure all dependencies are installed.")


if __name__ == "__main__":
    try:
        Config.validate()
        test_config()
        test_agent_initialization()
        
        print("\n✓ Configuration test completed successfully!")
        print("\nNext steps:")
        print("1. Review the configuration above")
        print("2. Modify .env file if needed")
        print("3. Run: python example_usage.py")
        
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease set up your .env file with required API keys.")
        print("See CONFIG_GUIDE.md for details.")
