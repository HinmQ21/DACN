"""Reasoning Agent - Suy luận logic dựa trên kiến thức y tế."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from utils.config import Config


class ReasoningAgent:
    """Agent suy luận y tế dựa trên kiến thức."""
    
    def __init__(self):
        # Use reasoning-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('reasoning'))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized physician with extensive medical knowledge.
Your task is to provide logical reasoning to answer medical questions based on:
1. Fundamental medical knowledge
2. Physiology and biochemistry
3. Pharmacology
4. Clinical diagnosis
5. Pathophysiology

Use step-by-step reasoning approach:
1. Analyze the question
2. Identify relevant foundational knowledge
3. Apply medical logic
4. Draw conclusions

Respond in this format:
ANALYSIS: [question analysis]
KNOWLEDGE: [relevant medical knowledge]
REASONING: [step-by-step reasoning process]
CONCLUSION: [final conclusion]"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def reason(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Perform logical reasoning.
        
        Args:
            question: Question to reason about
            context: Additional context (if any)
            
        Returns:
            Dictionary containing reasoning results
        """
        full_question = question
        if context:
            full_question = f"Question: {question}\n\nAdditional context: {context}"
        
        reasoning_result = self.chain.invoke({"question": full_question})
        
        # Parse structured output
        sections = {
            'analysis': '',
            'knowledge': '',
            'reasoning_steps': '',
            'conclusion': ''
        }
        
        lines = reasoning_result.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('ANALYSIS:'):
                current_section = 'analysis'
                sections['analysis'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('KNOWLEDGE:'):
                current_section = 'knowledge'
                sections['knowledge'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('REASONING:'):
                current_section = 'reasoning_steps'
                sections['reasoning_steps'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('CONCLUSION:'):
                current_section = 'conclusion'
                sections['conclusion'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif current_section and line.strip():
                sections[current_section] += ' ' + line.strip()
        
        return {
            'raw_output': reasoning_result,
            'structured': sections,
            'conclusion': sections['conclusion']
        }

