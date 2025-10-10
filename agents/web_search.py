"""Web Search Agent - Tìm kiếm thông tin từ Tavily và PubMed."""

from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
from Bio import Entrez
from utils.config import Config
import time


class WebSearchAgent:
    """Agent tìm kiếm web sử dụng Tavily và PubMed."""
    
    def __init__(self):
        # Use web_search-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('web_search'))
        
        # Tavily search tool
        self.tavily = TavilySearchResults(
            max_results=Config.MAX_SEARCH_RESULTS,
            api_key=Config.TAVILY_API_KEY
        )
        
        # PubMed configuration
        Entrez.email = "medical_agent@example.com"
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical expert who synthesizes information from search sources.
Your task is to analyze and synthesize information from search results.

Please:
1. Summarize important information
2. Evaluate source credibility
3. Extract evidence relevant to the question
4. Note any contradictions if present"""),
            ("human", "Question: {question}\n\nSearch results:\n{search_results}\n\nPlease synthesize the information.")
        ])
    
    def search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """Tìm kiếm sử dụng Tavily."""
        try:
            results = self.tavily.invoke({"query": query})
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    def search_pubmed(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """Tìm kiếm trên PubMed."""
        if max_results is None:
            max_results = Config.PUBMED_MAX_RESULTS
            
        try:
            # Search for articles
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            
            if not id_list:
                return []
            
            # Fetch article details
            time.sleep(0.5)  # Rate limiting
            handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            results = []
            for article in records.get('PubmedArticle', []):
                try:
                    medline = article['MedlineCitation']
                    title = medline['Article'].get('ArticleTitle', 'No title')
                    
                    abstract_parts = medline['Article'].get('Abstract', {}).get('AbstractText', [])
                    abstract = ' '.join([str(part) for part in abstract_parts])
                    
                    pmid = str(medline['PMID'])
                    
                    results.append({
                        'title': title,
                        'abstract': abstract,
                        'pmid': pmid,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                except Exception as e:
                    print(f"Error parsing PubMed article: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def search(self, question: str, key_terms: List[str] = None) -> Dict[str, Any]:
        """
        Tìm kiếm thông tin từ nhiều nguồn.
        
        Args:
            question: Câu hỏi cần tìm kiếm
            key_terms: Các từ khóa quan trọng
            
        Returns:
            Dictionary chứa kết quả tìm kiếm và tổng hợp
        """
        search_query = question
        if key_terms:
            search_query = " ".join(key_terms)
        
        # Parallel search
        tavily_results = self.search_tavily(search_query)
        pubmed_results = self.search_pubmed(search_query)
        
        # Format results for synthesis
        all_results = []
        
        for idx, result in enumerate(tavily_results[:3], 1):
            all_results.append(f"[Tavily {idx}] {result.get('content', '')}")
        
        for idx, result in enumerate(pubmed_results[:3], 1):
            all_results.append(
                f"[PubMed {idx}] Title: {result['title']}\n"
                f"Abstract: {result['abstract'][:500]}..."
            )
        
        # Synthesize information
        if all_results:
            synthesis_input = {
                "question": question,
                "search_results": "\n\n".join(all_results)
            }
            synthesis = (self.synthesis_prompt | self.llm).invoke(synthesis_input)
            synthesis_text = synthesis.content if hasattr(synthesis, 'content') else str(synthesis)
        else:
            synthesis_text = "No relevant information found."
        
        return {
            "tavily_results": tavily_results,
            "pubmed_results": pubmed_results,
            "synthesis": synthesis_text,
            "total_sources": len(tavily_results) + len(pubmed_results)
        }

