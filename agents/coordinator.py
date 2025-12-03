"""Coordinator Agent - Phân tích câu hỏi và điều phối workflow với Dynamic Few-shot Selection."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional
from utils.config import Config
import os


class CoordinatorAgent:
    """
    Agent điều phối - phân tích câu hỏi và quyết định chiến lược.
    Tích hợp Dynamic Few-shot Selection từ Medprompt.
    """
    
    def __init__(self, enable_few_shot: bool = None):
        """
        Initialize Coordinator Agent.
        
        Args:
            enable_few_shot: Whether to enable few-shot selection (uses config if None)
        """
        # Use coordinator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('coordinator'))
        
        # Few-shot configuration
        medprompt_config = Config.get_medprompt_config()
        self.enable_few_shot = enable_few_shot if enable_few_shot is not None else medprompt_config['enable_few_shot']
        self.few_shot_k = medprompt_config['few_shot_k']
        self.min_similarity = medprompt_config['few_shot_min_similarity']
        
        # Initialize KNN retriever (lazy loading)
        self._knn_retriever = None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coordinator agent in a multi-agent medical system.
Your task is to analyze medical questions and determine the search strategy.

Analyze the following factors:
1. Question type (diagnosis, treatment, physiology, pharmacology, etc.)
2. Question complexity
3. Key terms for searching
4. Whether web search is needed or reasoning alone is sufficient
5. Priority level between web search and reasoning

Return a concise analysis in JSON format:
{{
    "question_type": "question type",
    "complexity": "low/medium/high",
    "key_terms": ["keyword 1", "keyword 2"],
    "needs_web_search": true/false,
    "needs_reasoning": true/false,
    "search_priority": "high/medium/low",
    "reasoning_priority": "high/medium/low"
}}"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    @property
    def knn_retriever(self):
        """Lazy load KNN retriever."""
        if self._knn_retriever is None and self.enable_few_shot:
            try:
                from utils.knn_retriever import KNNRetriever
                index_path = Config.KNN_INDEX_PATH
                
                if os.path.exists(index_path):
                    self._knn_retriever = KNNRetriever(index_path=index_path)
                    print(f"[Coordinator] KNN retriever loaded from {index_path}")
                else:
                    print(f"[Coordinator] KNN index not found at {index_path}. Few-shot disabled.")
                    self.enable_few_shot = False
            except Exception as e:
                print(f"[Coordinator] Error loading KNN retriever: {e}. Few-shot disabled.")
                self.enable_few_shot = False
        
        return self._knn_retriever
    
    def get_similar_examples(
        self, 
        question: str, 
        options: Dict[str, str] = None,
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar examples using KNN.
        
        Args:
            question: The question to find similar examples for
            options: Question options (for multiple choice)
            k: Number of examples to retrieve (uses config if None)
            
        Returns:
            List of similar examples with similarity scores
        """
        if not self.enable_few_shot or self.knn_retriever is None:
            return []
        
        k = k or self.few_shot_k
        
        try:
            similar_examples = self.knn_retriever.get_similar_examples(
                query=question,
                k=k,
                options=options,
                include_options_in_query=True,
                min_similarity=self.min_similarity
            )
            return similar_examples
        except Exception as e:
            print(f"[Coordinator] Error retrieving similar examples: {e}")
            return []
    
    def format_few_shot_examples(
        self, 
        examples: List[Dict[str, Any]],
        include_reasoning: bool = True
    ) -> str:
        """
        Format few-shot examples for prompt inclusion.
        
        Args:
            examples: List of similar examples
            include_reasoning: Whether to include CoT reasoning
            
        Returns:
            Formatted string of examples
        """
        if not examples:
            return ""
        
        formatted_parts = []
        
        for i, ex in enumerate(examples, 1):
            parts = [f"Example {i}:"]
            
            # Question
            question = ex.get('question', 'N/A')
            parts.append(f"Question: {question}")
            
            # Options
            options = ex.get('options', {})
            if options:
                options_str = "\n".join([f"  {k}: {v}" for k, v in options.items()])
                parts.append(f"Options:\n{options_str}")
            
            # CoT Reasoning (if available)
            if include_reasoning:
                reasoning = ex.get('cot_reasoning', ex.get('reasoning', ''))
                if reasoning:
                    parts.append(f"Reasoning: {reasoning}")
            
            # Answer
            answer_idx = ex.get('answer_idx', '')
            answer = ex.get('answer', '')
            if answer_idx and answer:
                parts.append(f"Answer: {answer_idx} - {answer}")
            elif answer_idx:
                parts.append(f"Answer: {answer_idx}")
            elif answer:
                parts.append(f"Answer: {answer}")
            
            # Similarity score
            similarity = ex.get('similarity_score', 0)
            parts.append(f"(Similarity: {similarity:.3f})")
            
            formatted_parts.append("\n".join(parts))
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def analyze(
        self, 
        question: str,
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Phân tích câu hỏi và quyết định chiến lược.
        
        Args:
            question: Câu hỏi cần phân tích
            options: Các lựa chọn (nếu là multiple choice)
            
        Returns:
            Dictionary chứa phân tích và few-shot examples
        """
        # Get similar examples for few-shot learning
        similar_examples = self.get_similar_examples(question, options)
        
        # Analyze the question
        result = self.chain.invoke({"question": question})
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "question_type": "general",
                "complexity": "medium",
                "key_terms": [],
                "needs_web_search": True,
                "needs_reasoning": True,
                "search_priority": "medium",
                "reasoning_priority": "medium"
            }
        
        # Add few-shot examples to analysis
        analysis['few_shot_examples'] = similar_examples
        analysis['few_shot_examples_formatted'] = self.format_few_shot_examples(
            similar_examples, 
            include_reasoning=Config.ENABLE_COT
        )
        analysis['num_similar_examples'] = len(similar_examples)
        
        return analysis
    
    def analyze_with_context(
        self, 
        question: str,
        options: Dict[str, str] = None,
        additional_context: str = None
    ) -> Dict[str, Any]:
        """
        Analyze question with additional context.
        
        Args:
            question: The question to analyze
            options: Question options
            additional_context: Additional context to consider
            
        Returns:
            Analysis dictionary
        """
        # Build context-aware prompt
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coordinator agent in a multi-agent medical system.
Analyze the medical question considering the provided context and similar examples.

Your tasks:
1. Identify question type and complexity
2. Extract key medical terms
3. Determine if web search or reasoning is more important
4. Consider insights from similar examples

Return analysis in JSON format:
{{
    "question_type": "question type",
    "complexity": "low/medium/high", 
    "key_terms": ["keyword 1", "keyword 2"],
    "needs_web_search": true/false,
    "needs_reasoning": true/false,
    "search_priority": "high/medium/low",
    "reasoning_priority": "high/medium/low",
    "similar_examples_insight": "insights from similar examples"
}}"""),
            ("human", """Question: {question}

{context}

Please analyze this question.""")
        ])
        
        # Get similar examples
        similar_examples = self.get_similar_examples(question, options)
        formatted_examples = self.format_few_shot_examples(similar_examples)
        
        # Build context
        context_parts = []
        if formatted_examples:
            context_parts.append(f"Similar Examples:\n{formatted_examples}")
        if additional_context:
            context_parts.append(f"Additional Context:\n{additional_context}")
        
        context = "\n\n".join(context_parts) if context_parts else "No additional context."
        
        # Run analysis
        chain = context_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "context": context})
        
        # Parse result
        import json
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
        except json.JSONDecodeError:
            analysis = {
                "question_type": "general",
                "complexity": "medium",
                "key_terms": [],
                "needs_web_search": True,
                "needs_reasoning": True,
                "search_priority": "medium",
                "reasoning_priority": "medium"
            }
        
        # Add few-shot data
        analysis['few_shot_examples'] = similar_examples
        analysis['few_shot_examples_formatted'] = formatted_examples
        analysis['num_similar_examples'] = len(similar_examples)
        
        return analysis
