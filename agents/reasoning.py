"""Reasoning Agent - Suy luận logic dựa trên kiến thức y tế với Self-Generated CoT."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional
from utils.config import Config


class ReasoningAgent:
    """
    Agent suy luận y tế dựa trên kiến thức.
    Tích hợp Self-Generated Chain-of-Thought (CoT) từ Medprompt.
    """
    
    def __init__(self, enable_cot: bool = None):
        """
        Initialize Reasoning Agent.
        
        Args:
            enable_cot: Whether to enable detailed CoT (uses config if None)
        """
        # Use reasoning-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('reasoning'))
        
        # CoT configuration
        medprompt_config = Config.get_medprompt_config()
        self.enable_cot = enable_cot if enable_cot is not None else medprompt_config['enable_cot']
        self.cot_detailed = medprompt_config['cot_detailed']
        
        # Standard prompt (without few-shot examples)
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
        
        # CoT prompt with few-shot examples
        self.cot_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized physician with extensive medical knowledge.
Your task is to answer medical questions using step-by-step Chain-of-Thought reasoning.

Learn from these similar solved examples:

{few_shot_examples}

---

For the new question, follow this structured reasoning process:

STEP 1 - UNDERSTAND THE QUESTION:
- Identify the clinical scenario and patient presentation
- Note key symptoms, lab values, and history

STEP 2 - IDENTIFY KEY CONCEPTS:
- What medical knowledge domains are relevant?
- What are the key pathophysiological mechanisms?

STEP 3 - ANALYZE EACH OPTION:
- Systematically evaluate each answer choice
- Why might each option be correct or incorrect?

STEP 4 - APPLY CLINICAL REASONING:
- Use differential diagnosis approach
- Consider the most likely diagnosis/treatment/mechanism

STEP 5 - REACH CONCLUSION:
- State the final answer with confidence
- Summarize the key reasoning

Respond in this format:
ANALYSIS: [comprehensive question analysis]
KEY_CONCEPTS: [relevant medical concepts and knowledge]
OPTION_ANALYSIS: [analysis of each option]
REASONING: [detailed step-by-step reasoning]
CONCLUSION: [final answer with justification]"""),
            ("human", """Question: {question}

{options_text}

{additional_context}

Please provide detailed Chain-of-Thought reasoning.""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def reason(
        self, 
        question: str, 
        context: str = None,
        few_shot_examples: str = None,
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Perform logical reasoning with optional CoT and few-shot examples.
        
        Args:
            question: Question to reason about
            context: Additional context (web search results, etc.)
            few_shot_examples: Formatted few-shot examples from coordinator
            options: Answer options (for multiple choice)
            
        Returns:
            Dictionary containing reasoning results
        """
        # Decide whether to use CoT prompt
        use_cot = self.enable_cot and few_shot_examples
        
        if use_cot:
            return self._reason_with_cot(question, few_shot_examples, options, context)
        else:
            return self._reason_standard(question, context)
    
    def _reason_standard(self, question: str, context: str = None) -> Dict[str, Any]:
        """Standard reasoning without few-shot examples."""
        full_question = question
        if context:
            full_question = f"Question: {question}\n\nAdditional context: {context}"
        
        reasoning_result = self.chain.invoke({"question": full_question})
        
        return self._parse_reasoning_output(reasoning_result)
    
    def _reason_with_cot(
        self, 
        question: str, 
        few_shot_examples: str,
        options: Dict[str, str] = None,
        context: str = None
    ) -> Dict[str, Any]:
        """Reasoning with Chain-of-Thought and few-shot examples."""
        # Format options
        options_text = ""
        if options:
            options_text = "Options:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        # Format additional context
        additional_context = ""
        if context:
            additional_context = f"Additional Information:\n{context}"
        
        # Create chain with CoT prompt
        chain = self.cot_prompt | self.llm | StrOutputParser()
        
        reasoning_result = chain.invoke({
            "question": question,
            "few_shot_examples": few_shot_examples,
            "options_text": options_text,
            "additional_context": additional_context
        })
        
        return self._parse_cot_output(reasoning_result)
    
    def _parse_reasoning_output(self, result: str) -> Dict[str, Any]:
        """Parse standard reasoning output."""
        sections = {
            'analysis': '',
            'knowledge': '',
            'reasoning_steps': '',
            'conclusion': ''
        }
        
        lines = result.split('\n')
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
            'raw_output': result,
            'structured': sections,
            'conclusion': sections['conclusion'],
            'reasoning_type': 'standard'
        }
    
    def _parse_cot_output(self, result: str) -> Dict[str, Any]:
        """Parse CoT reasoning output."""
        sections = {
            'analysis': '',
            'key_concepts': '',
            'option_analysis': '',
            'reasoning_steps': '',
            'conclusion': ''
        }
        
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('ANALYSIS:'):
                current_section = 'analysis'
                sections['analysis'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('KEY_CONCEPTS:') or line_upper.startswith('KEY CONCEPTS:'):
                current_section = 'key_concepts'
                sections['key_concepts'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('OPTION_ANALYSIS:') or line_upper.startswith('OPTION ANALYSIS:'):
                current_section = 'option_analysis'
                sections['option_analysis'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('REASONING:'):
                current_section = 'reasoning_steps'
                sections['reasoning_steps'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('CONCLUSION:'):
                current_section = 'conclusion'
                sections['conclusion'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif current_section and line.strip():
                sections[current_section] += ' ' + line.strip()
        
        return {
            'raw_output': result,
            'structured': sections,
            'conclusion': sections['conclusion'],
            'option_analysis': sections['option_analysis'],
            'reasoning_type': 'cot'
        }
    
    def generate_cot_for_example(
        self, 
        question: str, 
        options: Dict[str, str],
        correct_answer: str
    ) -> str:
        """
        Generate Chain-of-Thought reasoning for a training example.
        Used for building the knowledge base with CoT.
        
        Args:
            question: The question
            options: Answer options
            correct_answer: The correct answer
            
        Returns:
            Generated CoT reasoning text
        """
        cot_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical education expert. Generate detailed Chain-of-Thought reasoning
for the following medical question. The reasoning should:
1. Explain the clinical scenario
2. Identify relevant medical concepts
3. Analyze why each option is correct or incorrect
4. Build a logical path to the correct answer

Be educational and thorough."""),
            ("human", """Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}

Generate detailed step-by-step reasoning that leads to this answer:""")
        ])
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        chain = cot_generation_prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "question": question,
            "options_text": options_text,
            "correct_answer": correct_answer
        })
        
        return result
    
    def reason_with_self_consistency(
        self,
        question: str,
        options: Dict[str, str] = None,
        few_shot_examples: str = None,
        num_samples: int = 3,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Perform reasoning with self-consistency (multiple sampling).
        
        Args:
            question: Question to reason about
            options: Answer options
            few_shot_examples: Few-shot examples
            num_samples: Number of reasoning samples to generate
            context: Additional context
            
        Returns:
            Aggregated reasoning results with consistency info
        """
        samples = []
        
        # Use higher temperature for diversity
        original_config = Config.get_llm_config('reasoning')
        diverse_llm = ChatGoogleGenerativeAI(
            **{**original_config, 'temperature': 0.7}
        )
        
        for i in range(num_samples):
            try:
                # Generate reasoning sample
                if self.enable_cot and few_shot_examples:
                    chain = self.cot_prompt | diverse_llm | StrOutputParser()
                    options_text = ""
                    if options:
                        options_text = "Options:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()])
                    
                    result = chain.invoke({
                        "question": question,
                        "few_shot_examples": few_shot_examples,
                        "options_text": options_text,
                        "additional_context": context or ""
                    })
                    parsed = self._parse_cot_output(result)
                else:
                    chain = self.prompt | diverse_llm | StrOutputParser()
                    full_question = question
                    if context:
                        full_question = f"Question: {question}\n\nAdditional context: {context}"
                    result = chain.invoke({"question": full_question})
                    parsed = self._parse_reasoning_output(result)
                
                samples.append(parsed)
                
            except Exception as e:
                print(f"[Reasoning] Error in sample {i+1}: {e}")
                continue
        
        if not samples:
            return self.reason(question, context, few_shot_examples, options)
        
        # Aggregate conclusions
        conclusions = [s['conclusion'] for s in samples if s.get('conclusion')]
        
        return {
            'raw_output': samples[0]['raw_output'],  # Use first sample as primary
            'structured': samples[0].get('structured', {}),
            'conclusion': conclusions[0] if conclusions else '',
            'all_conclusions': conclusions,
            'num_samples': len(samples),
            'reasoning_type': 'self_consistency'
        }
