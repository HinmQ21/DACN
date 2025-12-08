"""Image Agent - Medical image analysis and Visual Question Answering (VQA)."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Dict, Any, Optional, Union
from utils.config import Config
import base64
import os
import re
import urllib.request


class ImageAgent:
    """
    Agent phân tích ảnh y tế và trả lời câu hỏi dựa trên ảnh.
    
    Hỗ trợ:
    - Phân tích ảnh X-ray, CT, MRI, đơn thuốc, etc.
    - Visual Question Answering (VQA) cho ảnh y tế
    - Input từ file path hoặc URL
    """
    
    def __init__(self):
        """Initialize Image Agent with vision-capable model."""
        # Use image-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('image'))
        
        # System prompts
        self.analysis_system_prompt = """You are an expert medical image analyst with extensive experience in radiology, 
pathology, and clinical diagnostics. Your task is to analyze medical images and provide detailed, accurate findings.

IMPORTANT: Use this EXACT format for your response:

IMAGE TYPE: [Identify the type: X-ray, CT, MRI, ultrasound, prescription, etc.]

ANATOMICAL REGION: [Body part or region shown]

KEY FINDINGS:
- [Finding 1]
- [Finding 2]
- [Finding 3]
(List all significant observations, abnormalities, or notable features)

INTERPRETATION: [Clinical interpretation of the findings in 2-3 sentences]

RECOMMENDATIONS: [Follow-up actions or additional tests if applicable]

Note: This is an AI-assisted analysis and should be reviewed by a qualified healthcare professional.

Be thorough but concise. Use medical terminology appropriately."""

        self.vqa_system_prompt = """You are an expert medical image analyst specializing in answering questions about medical images.
Your task is to analyze the provided medical image and answer the specific question asked.

Guidelines:
1. For multiple choice: State the letter (A, B, C, or D) clearly at the start of your answer
2. Provide a clear explanation based on what you observe in the image
3. Use medical terminology appropriately
4. If uncertain, explain why
5. State your confidence level (high/medium/low)

Format your response clearly:
- If multiple choice: Start with "Answer: [Letter]" then explain
- If open-ended: Provide a direct answer then detailed explanation

Note: This is an AI-assisted analysis and should be reviewed by a qualified healthcare professional."""

    def _load_image(self, image_input: str) -> Dict[str, Any]:
        """
        Load image from file path or URL.
        
        Args:
            image_input: File path or URL to the image
            
        Returns:
            Dictionary with image data in LangChain format
        """
        # Check if input is URL
        if image_input.startswith(('http://', 'https://')):
            return self._load_from_url(image_input)
        else:
            return self._load_from_file(image_input)
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load image from local file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Determine MIME type
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Read and encode image
        with open(file_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        }
    
    def _load_from_url(self, url: str) -> Dict[str, Any]:
        """Load image from URL with proper headers to avoid 403 errors."""
        try:
            # Create request with User-Agent header to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            request = urllib.request.Request(url, headers=headers)
            
            # Download image
            with urllib.request.urlopen(request, timeout=30) as response:
                image_data = response.read()
            
            # Determine MIME type from URL or content-type
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            if 'png' in url.lower() or 'png' in content_type:
                mime_type = 'image/png'
            elif 'gif' in url.lower() or 'gif' in content_type:
                mime_type = 'image/gif'
            elif 'webp' in url.lower() or 'webp' in content_type:
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'
            
            # Encode as base64
            image_base64 = base64.standard_b64encode(image_data).decode('utf-8')
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        except Exception as e:
            # Fallback: pass URL directly and let LLM handle it
            print(f"[ImageAgent] Warning: Could not download image, passing URL directly: {e}")
            return {
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }
    
    def analyze_image(self, image_input: str) -> Dict[str, Any]:
        """
        Analyze a medical image and extract findings.
        
        Args:
            image_input: File path or URL to the medical image
            
        Returns:
            Dictionary containing:
            - image_type: Type of medical image
            - region: Anatomical region
            - findings: List of key findings
            - interpretation: Clinical interpretation
            - recommendations: Follow-up recommendations
            - raw_output: Full LLM response
            - confidence: Confidence level
        """
        try:
            # Load image
            image_content = self._load_image(image_input)
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.analysis_system_prompt},
                    image_content,
                    {"type": "text", "text": "Please analyze this medical image and provide detailed findings."}
                ]
            )
            
            # Get response
            response = self.llm.invoke([message])
            raw_output = response.content
            
            # Parse response
            parsed = self._parse_analysis_output(raw_output)
            parsed['raw_output'] = raw_output
            parsed['image_source'] = image_input
            parsed['success'] = True
            
            return parsed
            
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': str(e),
                'image_source': image_input
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error analyzing image: {str(e)}",
                'image_source': image_input
            }
    
    def answer_question(
        self, 
        image_input: str, 
        question: str,
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about a medical image (VQA).
        
        Args:
            image_input: File path or URL to the medical image
            question: Question to answer about the image
            options: Answer options for multiple choice (optional)
            
        Returns:
            Dictionary containing:
            - answer: The answer to the question
            - explanation: Explanation of the answer
            - confidence: Confidence level (high/medium/low)
            - raw_output: Full LLM response
        """
        try:
            # Load image
            image_content = self._load_image(image_input)
            
            # Format question with options if provided
            question_text = f"Question: {question}"
            if options:
                options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
                question_text += f"\n\nOptions:\n{options_text}"
                question_text += "\n\nPlease select the correct answer and explain your reasoning."
            
            # Create message with image and question
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.vqa_system_prompt},
                    image_content,
                    {"type": "text", "text": question_text}
                ]
            )
            
            # Get response
            response = self.llm.invoke([message])
            raw_output = response.content
            
            # Parse response
            parsed = self._parse_vqa_output(raw_output, options)
            parsed['raw_output'] = raw_output
            parsed['image_source'] = image_input
            parsed['question'] = question
            parsed['success'] = True
            
            return parsed
            
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': str(e),
                'image_source': image_input,
                'question': question
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error answering question: {str(e)}",
                'image_source': image_input,
                'question': question
            }
    
    def _parse_analysis_output(self, output: str) -> Dict[str, Any]:
        """Parse the analysis output into structured format with flexible parsing."""
        result = {
            'image_type': '',
            'region': '',
            'findings': [],
            'interpretation': '',
            'recommendations': '',
            'confidence': 'medium'
        }
        
        # Remove markdown formatting
        output = re.sub(r'\*\*', '', output)  # Remove bold markers
        output = re.sub(r'^\s*#+\s*', '', output, flags=re.MULTILINE)  # Remove headers
        
        lines = output.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            line_upper = line_stripped.upper()
            
            # Detect section headers (more flexible matching)
            if any(keyword in line_upper for keyword in ['IMAGE TYPE:', '1. IMAGE TYPE', '1.IMAGE TYPE', 'TYPE OF IMAGE']):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = 'image_type'
                # Extract content after colon or header
                content = re.split(r'[:]\s*', line_stripped, maxsplit=1)
                current_content = [content[1] if len(content) > 1 else '']
                
            elif any(keyword in line_upper for keyword in ['ANATOMICAL REGION:', '2. ANATOMICAL', 'REGION:', 'BODY REGION']):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = 'region'
                content = re.split(r'[:]\s*', line_stripped, maxsplit=1)
                current_content = [content[1] if len(content) > 1 else '']
                
            elif any(keyword in line_upper for keyword in ['KEY FINDINGS:', '3. KEY FINDINGS', 'FINDINGS:', 'OBSERVATIONS']):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = 'findings'
                content = re.split(r'[:]\s*', line_stripped, maxsplit=1)
                current_content = [content[1]] if len(content) > 1 and content[1] else []
                
            elif any(keyword in line_upper for keyword in ['INTERPRETATION:', '4. INTERPRETATION', 'CLINICAL INTERPRETATION']):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = 'interpretation'
                content = re.split(r'[:]\s*', line_stripped, maxsplit=1)
                current_content = [content[1] if len(content) > 1 else '']
                
            elif any(keyword in line_upper for keyword in ['RECOMMENDATION', '5. RECOMMENDATION', 'FOLLOW-UP', 'NEXT STEPS']):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = 'recommendations'
                content = re.split(r'[:]\s*', line_stripped, maxsplit=1)
                current_content = [content[1] if len(content) > 1 else '']
                
            elif current_section:
                # Add to current section content
                current_content.append(line_stripped)
        
        # Save last section
        if current_section and current_content:
            self._save_section(result, current_section, current_content)
        
        # If no structured data found, try to extract from full text
        if not result['image_type'] and not result['findings']:
            result['interpretation'] = output.strip()
            result['image_type'] = 'Medical Image'
        
        return result
    
    def _save_section(self, result: Dict, section: str, content: list):
        """Save parsed section content to result dict."""
        if section == 'findings':
            # Parse findings as list
            findings = []
            for line in content:
                # Remove bullet points, numbering, and extra formatting
                cleaned = re.sub(r'^[-•*+]\s*', '', line)
                cleaned = re.sub(r'^\d+[.)]\s*', '', cleaned)
                cleaned = cleaned.strip()
                # Skip empty lines and common phrases
                if cleaned and cleaned not in ['--', '...', 'None', 'N/A']:
                    findings.append(cleaned)
            result['findings'] = findings
        else:
            # Join other sections as text, filter out empty strings
            text = ' '.join([c.strip() for c in content if c.strip()])
            result[section] = text
    
    def _parse_vqa_output(self, output: str, options: Dict[str, str] = None) -> Dict[str, Any]:
        """Parse the VQA output into structured format with better extraction."""
        result = {
            'answer': '',
            'explanation': '',
            'confidence': 'medium'
        }
        
        # Remove markdown formatting
        clean_output = re.sub(r'\*\*', '', output)
        
        # Try to extract answer for multiple choice
        if options:
            # Look for answer pattern like "A", "Answer: A", "The answer is A"
            answer_patterns = [
                r'Answer:\s*([A-Z])\b',
                r'(?:correct answer|the answer is)[:\s]*([A-Z])\b',
                r'^([A-Z])[.:)\s]',
                r'\b([A-Z])\s*(?:is correct|is the correct)',
                r'(?:option|choice)\s*([A-Z])\b',
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, clean_output, re.IGNORECASE | re.MULTILINE)
                if match:
                    answer_letter = match.group(1).upper()
                    if answer_letter in options:
                        result['answer'] = answer_letter
                        break
            
            # If no answer found, try to find the option letter mentioned most prominently
            if not result['answer']:
                # Count mentions of each option at the beginning of the text
                first_500_chars = clean_output[:500]
                for key in sorted(options.keys()):  # Sort to ensure consistent order
                    if re.search(rf'\b{key}\b', first_500_chars):
                        result['answer'] = key
                        break
        
        # If still no answer and it's multiple choice, try to extract from full text
        if options and not result['answer']:
            # Last resort: find first capital letter that's an option
            for char in clean_output:
                if char.isupper() and char in options:
                    result['answer'] = char
                    break
        
        # Extract explanation (remove the answer line if present)
        explanation_lines = []
        for line in output.split('\n'):
            # Skip lines that only contain "Answer: X"
            if not re.match(r'^\s*Answer:\s*[A-Z]\s*$', line, re.IGNORECASE):
                explanation_lines.append(line)
        result['explanation'] = '\n'.join(explanation_lines).strip() or output
        
        # Extract confidence
        lower_output = output.lower()
        if any(phrase in lower_output for phrase in ['high confidence', 'very confident', 'definitely', 'clearly']):
            result['confidence'] = 'high'
        elif any(phrase in lower_output for phrase in ['low confidence', 'uncertain', 'unclear', 'difficult to']):
            result['confidence'] = 'low'
        else:
            result['confidence'] = 'medium'
        
        return result

