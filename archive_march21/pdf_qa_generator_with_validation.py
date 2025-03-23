import os
import re
import math
import base64
import argparse
import time
from typing import List, Dict, Tuple, Optional, Any
import tempfile
import anthropic
from anthropic.types import MessageParam
import fitz  # PyMuPDF
from PIL import Image
import io
import json

explanation = """
	explanation: 
		A: Self-contained explanation of why A is right/wrong without hinting at other options,
		B: Self-contained explanation of why B is right/wrong without hinting at other options,
		C: Self-contained explanation of why C is right/wrong without hinting at other options,
		D: Self-contained explanation of why D is right/wrong without hinting at other options,	
"""

class MCQValidator:
    """
    Validator for Multiple Choice Questions generated from PDFs.
    Checks if questions follow specified formatting and content guidelines.
    """
    
    def __init__(self):
        # Forbidden phrases in incorrect explanations
        self.forbidden_phrases = [
            "unlike the correct answer",
            "the correct answer is",
            "this confuses",
            "provided by the correct answer",
            "this is not what the question is asking for",
            "this does not address the question",
            "this would be correct if",
            "unlike other options",
            "compared to",
            "rather than",
            "instead of"
        ]
        
        # Presentation reference phrases to avoid
        self.presentation_references = [
            "according to the slides",
            "in the presentation",
            "in the material",
            "as stated in presentation",
            "the presentation",
            "as shown on page",
            "on page",
            "in the pdf",
            "in the document"
        ]
        
    def extract_questions(self, text: str) -> List[Dict]:
        """
        Extract individual questions and their components from the text.
        
        Args:
            text: The full text response from Claude
            
        Returns:
            List of dictionaries containing parsed questions
        """
        # Print the first 500 characters for debugging
        print(f"\nAnalyzing text (first 200 chars): {text[:200]}...\n")
        
        # First try to find questions using the standard format
        questions_pattern = r'(?:^|\n)Question (\d+)(?: \(Page[s]? ([\d-]+)\))?:(.*?)(?=(?:\n(?:Question \d+|$))|\Z)'
        question_matches = re.findall(questions_pattern, text, re.DOTALL)
        
        # If standard pattern doesn't work, try alternative formats
        if not question_matches:
            print("Standard question pattern not found, trying alternatives...")
            
            # Try a more relaxed pattern
            questions_pattern = r'(?:^|\n)(?:Question|Q)\.? (\d+)(?: \(Page[s]? ([\d-]+)\))?:(.*?)(?=(?:\n(?:Question|Q)\.? \d+|$))|\Z'
            question_matches = re.findall(questions_pattern, text, re.DOTALL)
            
            # If still no matches, split on Question headers
            if not question_matches:
                print("Alternative pattern not found, trying to split by headers...")
                question_blocks = re.split(r'(?:^|\n)Question \d+', text)[1:]  # Skip intro text
                
                if not question_blocks:
                    # One more try with a very relaxed pattern
                    question_pattern = r'(?:^|\n\n)((?:Question|Q)\.? \d+.*?)(?:\n\n(?:Question|Q)\.? \d+|$)'
                    question_blocks = re.findall(question_pattern, text, re.DOTALL)
                    
                # Process blocks if we found them
                if question_blocks:
                    parsed_questions = []
                    for i, block in enumerate(question_blocks):
                        parsed_question = self._parse_question_block(i+1, block)
                        parsed_questions.append(parsed_question)
                    return parsed_questions
                    
        # Process matched questions if found with regex pattern
        if question_matches:
            parsed_questions = []
            for i, match in enumerate(question_matches):
                q_num, page_nums, q_content = match
                page_numbers = page_nums if page_nums else "Unknown"
                
                # Parse the content for this question
                parsed_question = self._parse_question_content(i+1, page_numbers, q_content)
                parsed_questions.append(parsed_question)
                
            return parsed_questions
            
        # Fallback: just try to find anything that looks remotely like a question
        print("Warning: No questions found with standard patterns. Using fallback parsing.")
        fallback_pattern = r'(?:^|\n)(.*?\?|.*?:)(?:\n[A-D]\..*){2,}'
        fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
        
        if fallback_matches:
            parsed_questions = []
            for i, content in enumerate(fallback_matches):
                print(f"Attempting to parse fallback question #{i+1}")
                parsed_question = self._parse_question_content(i+1, "Unknown", content)
                parsed_questions.append(parsed_question)
            return parsed_questions
            
        # If all else fails, return a single error entry
        print("Error: Failed to extract any questions from text. Check the format.")
        return [{
            "question_num": 1,
            "error": "Failed to extract any questions from text",
            "raw_text": text[:1000] + ("..." if len(text) > 1000 else "")
        }]
    
    def _parse_question_block(self, q_num: int, block: str) -> Dict:
        """
        Parse a block of text containing a question.
        
        Args:
            q_num: Question number
            block: Block of text containing the question
            
        Returns:
            Dict with parsed question data
        """
        try:
            # Extract page numbers
            page_match = re.search(r'\(Page[s]? ([\d-]+)\)', block)
            page_numbers = page_match.group(1) if page_match else "Unknown"
            
            # Let the standard parser handle the content
            return self._parse_question_content(q_num, page_numbers, block)
            
        except Exception as e:
            return {
                "question_num": q_num,
                "error": f"Failed to parse question block: {str(e)}",
                "raw_text": block
            }
    
    def _parse_question_content(self, q_num: int, page_numbers: str, content: str) -> Dict:
        """
        Parse the content of a question for options and explanations.
        
        Args:
            q_num: Question number
            page_numbers: Page numbers string
            content: Question content
            
        Returns:
            Dict with parsed question data
        """
        try:
            # Extract question text - everything up to first option
            question_text = content.split('A.')[0].strip() if 'A.' in content else content.strip()
            
            # Clean up question text if it contains the page numbers
            if f"(Page{'' if page_numbers.isdigit() else 's'} {page_numbers}):" in question_text:
                question_text = question_text.split(f"(Page{'' if page_numbers.isdigit() else 's'} {page_numbers}):", 1)[1].strip()
            
            # Extract options
            options = {}
            option_pattern = r'([A-D])\. (.*?)(?=(?:[A-D]\.|\n\n|Explanation:|$))'
            option_matches = re.findall(option_pattern, content, re.DOTALL)
            
            for letter, text in option_matches:
                options[letter] = text.strip()
            
            # Extract explanations
            explanations = {}
            explanation_section = re.search(r'Explanation:(.*?)(?:\n\n|$)', content, re.DOTALL)
            
            if explanation_section:
                explanation_text = explanation_section.group(1).strip()
                
                # Try pattern with letter followed by period
                exp_pattern = r'([A-D])\. ((?:Correct|Incorrect):[^A-D\.]*(?:\n(?![A-D]\.).*)*)'
                explanation_matches = re.findall(exp_pattern, explanation_text, re.DOTALL)
                
                # If no matches, try alternative patterns
                if not explanation_matches:
                    # Try without period after letter
                    alt_pattern = r'([A-D]) ((?:Correct|Incorrect):[^A-D]*(?:\n(?![A-D]).*)*)'
                    explanation_matches = re.findall(alt_pattern, explanation_text, re.DOTALL)
                
                for letter, text in explanation_matches:
                    explanations[letter] = text.strip()
            
            # If no explanations found with regex, try a fallback approach
            if not explanations and explanation_section:
                explanation_text = explanation_section.group(1).strip()
                lines = explanation_text.split('\n')
                current_letter = None
                current_text = ""
                
                for line in lines:
                    letter_match = re.match(r'^\s*([A-D])[\.:]\s*(.*)', line)
                    if letter_match:
                        # Save previous letter if exists
                        if current_letter and current_text:
                            explanations[current_letter] = current_text.strip()
                        
                        # Start new letter
                        current_letter = letter_match.group(1)
                        current_text = letter_match.group(2)
                    elif current_letter:
                        # Continue previous explanation
                        current_text += "\n" + line
                
                # Save the last explanation
                if current_letter and current_text:
                    explanations[current_letter] = current_text.strip()
            
            return {
                "question_num": q_num,
                "page_numbers": page_numbers,
                "question_text": question_text,
                "options": options,
                "explanations": explanations
            }
            
        except Exception as e:
            return {
                "question_num": q_num,
                "error": f"Failed to parse question content: {str(e)}",
                "raw_text": content[:500] + ("..." if len(content) > 500 else "")
            }
    
    def validate_question(self, question: Dict) -> List[str]:
        """
        Validate a single question against all requirements.
        
        Args:
            question: Dictionary containing parsed question
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check if there was a parsing error
        if "error" in question:
            issues.append(f"Question {question['question_num']}: Parsing error - {question['error']}")
            return issues
            
        # Check if page numbers are included
        if question["page_numbers"] == "Unknown":
            issues.append(f"Question {question['question_num']}: Missing page numbers")
            
        # Check if question has 4 options (A, B, C, D)
        expected_options = set(['A', 'B', 'C', 'D'])
        actual_options = set(question["options"].keys())
        if actual_options != expected_options:
            missing = expected_options - actual_options
            extra = actual_options - expected_options
            if missing:
                issues.append(f"Question {question['question_num']}: Missing options: {', '.join(missing)}")
            if extra:
                issues.append(f"Question {question['question_num']}: Unexpected options: {', '.join(extra)}")
                
        # Check if explanations exist for all options
        for opt in expected_options:
            if opt not in question["explanations"]:
                issues.append(f"Question {question['question_num']}: Missing explanation for option {opt}")
                
        # Check if explanations start with "Correct:" or "Incorrect:"
        for opt, explanation in question["explanations"].items():
            if not explanation.startswith("Correct:") and not explanation.startswith("Incorrect:"):
                issues.append(f"Question {question['question_num']}: Explanation for option {opt} doesn't start with 'Correct:' or 'Incorrect:'")
                
        # Count correct answers
        correct_count = sum(1 for exp in question["explanations"].values() if exp.startswith("Correct:"))
        if correct_count != 1:
            issues.append(f"Question {question['question_num']}: Found {correct_count} correct answers, should be exactly 1")
                
        # Check for forbidden phrases in incorrect explanations
        for opt, explanation in question["explanations"].items():
            if explanation.startswith("Incorrect:"):
                lower_exp = explanation.lower()
                
                # Check for forbidden phrases
                for phrase in self.forbidden_phrases:
                    if phrase.lower() in lower_exp:
                        issues.append(f"Question {question['question_num']}: Explanation for option {opt} contains forbidden phrase: '{phrase}'")
                        
                # Check for presentation references
                for phrase in self.presentation_references:
                    if phrase.lower() in lower_exp:
                        issues.append(f"Question {question['question_num']}: Explanation for option {opt} references the presentation with phrase: '{phrase}'")
                
                # Check for page references in explanations
                if re.search(r'page \d+', lower_exp) or re.search(r'pages \d+', lower_exp):
                    issues.append(f"Question {question['question_num']}: Explanation for option {opt} references page numbers")
                
        return issues
    
    def validate_questions(self, text: str) -> Dict:
        """
        Validate all questions in the provided text.
        
        Args:
            text: The full text response from Claude
            
        Returns:
            Dictionary with validation results
        """
        questions = self.extract_questions(text)
        
        all_issues = []
        questions_with_issues = 0
        
        for question in questions:
            issues = self.validate_question(question)
            if issues:
                all_issues.extend(issues)
                questions_with_issues += 1
                
        return {
            "total_questions": len(questions),
            "questions_with_issues": questions_with_issues,
            "issues": all_issues,
            "parsed_questions": questions
        }
    
    def suggest_corrections(self, validation_results: Dict) -> str:
        """
        Generate suggestions to fix validation issues.
        
        Args:
            validation_results: Dictionary with validation results
            
        Returns:
            String with correction suggestions
        """
        if not validation_results["issues"]:
            return "All questions follow the specified format and requirements. No corrections needed."
            
        suggestions = [
            f"Found {len(validation_results['issues'])} issues in {validation_results['questions_with_issues']} of {validation_results['total_questions']} questions:",
            ""
        ]
        
        # Group issues by question
        issues_by_question = {}
        for issue in validation_results["issues"]:
            match = re.match(r'Question (\d+):', issue)
            if match:
                q_num = match.group(1)
                if q_num not in issues_by_question:
                    issues_by_question[q_num] = []
                issues_by_question[q_num].append(issue)
                
        # Generate suggestions for each question with issues
        for q_num, issues in sorted(issues_by_question.items(), key=lambda x: int(x[0])):
            suggestions.append(f"Question {q_num}:")
            for issue in issues:
                # Remove the "Question X:" prefix for cleaner display
                issue_text = re.sub(r'^Question \d+: ', '', issue)
                suggestions.append(f"- {issue_text}")
                
                # Add specific correction advice based on issue type
                if "doesn't start with 'Correct:' or 'Incorrect:'" in issue:
                    suggestions.append("  Correction: Make sure each explanation begins with either 'Correct:' or 'Incorrect:'")
                    
                elif "contains forbidden phrase" in issue:
                    suggestions.append("  Correction: Rewrite without comparing to other options or referencing the correct answer")
                    
                elif "references the presentation" in issue:
                    suggestions.append("  Correction: Present information as established knowledge without referencing the presentation materials")
                    
                elif "references page numbers" in issue:
                    suggestions.append("  Correction: Remove page number references from explanations")
                    
                elif "Found" in issue and "correct answers" in issue:
                    suggestions.append("  Correction: Ensure exactly one option is marked as 'Correct:' and the rest as 'Incorrect:'")
                    
            suggestions.append("")
            
        suggestions.append("General guidelines for corrections:")
        suggestions.append("1. Ensure each explanation stands completely on its own")
        suggestions.append("2. For incorrect options, focus only on why that specific option is wrong")
        suggestions.append("3. Never hint at or reference the correct answer in explanations for incorrect options")
        suggestions.append("4. Avoid comparative language like 'rather than' or 'instead of'")
        suggestions.append("5. Never include phrases like 'this doesn't address the question'")
        
        return "\n".join(suggestions)

    def generate_correction_prompt(self, validation_results: Dict, questions_text: str) -> str:
        """
        Generate a prompt to send back to Claude for correcting issues.
        
        Args:
            validation_results: Validation results dictionary
            questions_text: Original questions text
            
        Returns:
            Correction prompt text
        """
        if not validation_results["issues"]:
            return "All questions follow the required format. No corrections needed."
        
        prompt = [
            "I'm reviewing the MCQs you generated and found these issues that need correction:",
            ""
        ]
        
        # Add issues grouped by question
        issues_by_question = {}
        for issue in validation_results["issues"]:
            match = issue.split(":", 1)
            if len(match) == 2:
                q_label, issue_text = match
                if q_label not in issues_by_question:
                    issues_by_question[q_label] = []
                issues_by_question[q_label].append(issue_text.strip())
        
        for q_label, issues in issues_by_question.items():
            prompt.append(f"{q_label}:")
            for i, issue in enumerate(issues, 1):
                prompt.append(f"  {i}. {issue}")
            prompt.append("")
        
        # Add general guidance
        prompt.append("Please update these questions to follow the guidelines exactly:")
        prompt.append("- Begin EVERY option explanation with \"Correct:\" or \"Incorrect:\"")
        prompt.append("- For INCORRECT options, never mention or hint at the correct answer")
        prompt.append("- Never use comparative language like \"rather than\" or \"instead of\"")
        prompt.append("- Never reference page numbers in any explanation")
        prompt.append("- Each explanation must stand completely on its own")
        prompt.append("- Never use phrases like \"this doesn't address the question\"")
        prompt.append("- Present information as established knowledge in the field")
        prompt.append("")
        prompt.append("Please provide ONLY the corrected questions and explanations in the same format, without any additional text.")
        prompt.append("")
        prompt.append("Here are the original questions that need correction:")
        prompt.append("```")
        prompt.append(questions_text)
        prompt.append("```")
        
        return "\n".join(prompt)


class PDFVisionProcessor:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the PDF vision processor with API key and model.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use (must be a vision-capable model)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        # Use Claude 3.7 Sonnet by default
        self.model = model
        self.summaries = []
        self.questions = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.validator = MCQValidator()

    def extract_pages_as_images(self, file_path: str, skip_pages: int = 3, dpi: int = 300, 
                             optimize_tokens: bool = True) -> Tuple[List[Dict], int]:
        """
        Extract pages from a PDF as images.
        
        Args:
            file_path: Path to the PDF file
            skip_pages: Number of initial pages to skip (default: 3)
            dpi: Resolution for page rendering (default: 300)
            optimize_tokens: Whether to optimize images for token usage (default: True)
            
        Returns:
            Tuple of (list_of_page_image_data, effective_page_count)
        """
        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)
            
            # Skip the first n pages
            effective_start_page = min(skip_pages, total_pages)
            effective_pages = total_pages - effective_start_page
            
            print(f"PDF has {total_pages} total pages. Skipping first {skip_pages} pages.")
            print(f"Will process {effective_pages} pages (pages {effective_start_page+1} to {total_pages}).")
            
            # Extract each page as an image
            pages_data = []
            
            for page_num in range(effective_start_page, total_pages):
                # Get the page
                page = pdf_document.load_page(page_num)
                
                # Calculate image resolution - use lower DPI if optimizing for tokens
                page_dpi = min(dpi, 150) if optimize_tokens else dpi
                
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(page_dpi/72, page_dpi/72))
                img_data = pix.tobytes()
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB mode if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Optimize image if requested
                if optimize_tokens:
                    # Resize if larger than 1000 pixels in any dimension
                    max_dimension = 1000
                    if img.width > max_dimension or img.height > max_dimension:
                        ratio = min(max_dimension / img.width, max_dimension / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to base64
                buffered = io.BytesIO()
                # Use JPEG with reduced quality for token optimization
                quality = 75 if optimize_tokens else 90
                img.save(buffered, format="JPEG", quality=quality)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Estimate token count (very rough estimate: ~125 tokens per KB of base64)
                estimated_tokens = len(img_base64) // 750
                
                pages_data.append({
                    "page_num": page_num + 1,  # 1-indexed page number
                    "image_data": img_base64,
                    "estimated_tokens": estimated_tokens
                })
                
                print(f"Processed page {page_num + 1} as image - Est. tokens: {estimated_tokens}")
            
            pdf_document.close()
            return pages_data, effective_pages
            
        except Exception as e:
            print(f"Error extracting pages as images: {e}")
            return [], 0

    def chunk_pdf_pages(self, pages_data: List[Dict], total_pages: int, target_pages_per_chunk: int = 30) -> List[List[Dict]]:
        """
        Divide the PDF pages into chunks.
        
        Args:
            pages_data: List of page image data dictionaries
            total_pages: Total number of effective pages
            target_pages_per_chunk: Target number of pages per chunk
            
        Returns:
            List of chunks, where each chunk is a list of page data dictionaries
        """
        # If total pages is close to target, keep as single chunk
        if total_pages <= target_pages_per_chunk + 5:
            return [pages_data]
        
        # Calculate optimal number of chunks
        num_chunks = max(1, round(total_pages / target_pages_per_chunk))
        
        # If dividing would make chunks too small, reduce number of chunks
        pages_per_chunk = total_pages / num_chunks
        if pages_per_chunk < 15 and num_chunks > 1:
            num_chunks -= 1
            pages_per_chunk = total_pages / num_chunks
        
        # Calculate pages per chunk
        pages_per_chunk = math.ceil(len(pages_data) / num_chunks)
        
        # Divide into chunks
        chunks = []
        for i in range(0, len(pages_data), pages_per_chunk):
            chunk = pages_data[i:i + pages_per_chunk]
            chunks.append(chunk)
        
        print(f"Divided {total_pages} pages into {len(chunks)} chunks " +
              f"(~{total_pages/len(chunks):.1f} pages per chunk)")
            
        return chunks

    def get_page_range(self, chunk: List[Dict]) -> Tuple[int, int]:
        """
        Get the first and last page number from a chunk of pages.
        
        Args:
            chunk: List of page data dictionaries
            
        Returns:
            Tuple of (first_page, last_page)
        """
        if not chunk:
            return 0, 0
            
        return chunk[0]["page_num"], chunk[-1]["page_num"]

    def generate_first_chunk_prompt(self, chunk: List[Dict], total_chunks: int) -> Tuple[str, List[Dict]]:
        """
        Generate prompt for the first chunk, instructing to avoid irrelevant content.
        
        Args:
            chunk: First chunk of PDF pages
            total_chunks: Total number of chunks
            
        Returns:
            Tuple of (prompt_text, message_content)
        """
        first_page, last_page = self.get_page_range(chunk)
        
        # If only one chunk exists, generate 10 questions, otherwise 5
        num_questions = 10 if total_chunks == 1 else 5
        
        prompt = f"""You are a subject matter expert creating MCQs from pages {first_page} to {last_page}. Generate {num_questions} challenging multiple-choice questions.

## CONTENT REQUIREMENTS
- Focus ONLY on educational content (ignore title slides, agenda slides, etc.)
- Pay special attention to tables, graphs, diagrams, and visual elements
- Each question MUST have 4 options (A,B,C,D) with exactly ONE correct answer
- All questions and options MUST come DIRECTLY from the PDF content
- Include page numbers beside questions (e.g., "Question 1 (Page 4)")
- If content spans multiple pages, include range (e.g., "Question 1 (Pages 4-6)")
- Use action verbs in questions when appropriate
- Ensure proper grammar in all questions and options
- AVOID using "NOT" in questions or answer options
- "All of the above" can be used as an option when appropriate

## CRITICAL EXPLANATION RULES
- Begin EVERY option explanation with "Correct:" or "Incorrect:"
- For INCORRECT options:
  * Focus ONLY on factual content of that specific option
  * Explain why it's wrong using ONLY content from the PDF & keep it short
  * Pretend you don't know which option is correct
  * NEVER mention, reference, or hint at the correct answer
  * NEVER use comparative language ("rather than", "instead of")
  * NEVER use phrases like "this doesn't address the question" or "would be correct if..."
- Each explanation MUST stand completely on its own
- NEVER invent or introduce information not present in the PDF
- NEVER use phrases like "according to the slides," "in the presentation," "in the material," etc.
- NEVER reference page numbers in any explanation (e.g., "as shown on page X," "on page 5," etc.)
- Present information as established knowledge in the field

## FORBIDDEN PHRASES IN INCORRECT EXPLANATIONS
- "unlike the correct answer"
- "the correct answer is"
- "this confuses X with Y"
- "provided by the correct answer"
- "this is not what the question is asking for"
- "this does not address the question"
- "this would be correct if..."
- "unlike other options"
- "compared to"
- "rather than"
- "instead of"

## OUTPUT FORMAT
```
Question 1 (Page X): [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

Explanation:
A. [Correct/Incorrect]: [Self-contained explanation about only this option]
B. [Correct/Incorrect]: [Self-contained explanation about only this option]
C. [Correct/Incorrect]: [Self-contained explanation about only this option]
D. [Correct/Incorrect]: [Self-contained explanation about only this option]

Question 2 (Page X): [Question text]
...
```

After all questions, provide a brief 2-3 sentence summary of key concepts covered.
"""
        
        # Construct message content with both text and images
        message_content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add images for each page
        for page_data in chunk:
            message_content.append({
                "type": "image", 
                "source": {
                    "type": "base64", 
                    "media_type": "image/jpeg", 
                    "data": page_data["image_data"]
                }
            })
            
        return prompt, message_content

    def generate_subsequent_chunk_prompt(self, chunk: List[Dict], previous_summaries: str) -> Tuple[str, List[Dict]]:
        """
        Generate prompt for subsequent chunks, including previous summaries for context.
        
        Args:
            chunk: Current chunk of PDF pages
            previous_summaries: Combined summaries from previous chunks
            
        Returns:
            Tuple of (prompt_text, message_content)
        """
        first_page, last_page = self.get_page_range(chunk)
        
        num_questions = 5  # Default for subsequent chunks
        
        prompt = f"""You are a subject matter expert creating MCQs from pages {first_page} to {last_page}. Generate {num_questions} challenging multiple-choice questions focused on NEW concepts in this section.

## CONTEXT FROM PREVIOUS SECTIONS
{previous_summaries}

## CONTENT REQUIREMENTS
- Focus ONLY on educational content (ignore title slides, agenda slides, etc.)
- Pay special attention to tables, graphs, diagrams, and visual elements
- Each question MUST have 4 options (A,B,C,D) with exactly ONE correct answer
- All questions and options MUST come DIRECTLY from the PDF content
- Include page numbers beside questions (e.g., "Question 1 (Page 4)")
- If content spans multiple pages, include range (e.g., "Question 1 (Pages 4-6)")
- Use action verbs in questions when appropriate
- Ensure proper grammar in all questions and options
- AVOID using "NOT" in questions or answer options
- "All of the above" can be used as an option when appropriate

## CRITICAL EXPLANATION RULES
- Begin EVERY option explanation with "Correct:" or "Incorrect:"
- For INCORRECT options:
  * Focus ONLY on factual content of that specific option
  * Explain why it's wrong using ONLY content from the PDF & keep it short
  * Pretend you don't know which option is correct
  * NEVER mention, reference, or hint at the correct answer
  * NEVER use comparative language ("rather than", "instead of")
  * NEVER use phrases like "this doesn't address the question" or "would be correct if..."
- Each explanation MUST stand completely on its own
- NEVER invent or introduce information not present in the PDF
- NEVER use phrases like "according to the slides," "in the presentation," "in the material," "As stated in presentation," "The presentation," etc.
- NEVER reference page numbers in any explanation (e.g., "as shown on page X," "on page 5," etc.)
- Present information as established knowledge in the field

## FORBIDDEN PHRASES IN INCORRECT EXPLANATIONS
- "unlike the correct answer"
- "the correct answer is"
- "this confuses X with Y"
- "provided by the correct answer"
- "this is not what the question is asking for"
- "this does not address the question"
- "this would be correct if..."
- "unlike other options"
- "compared to"
- "rather than"
- "instead of"

## OUTPUT FORMAT
```
Question 1 (Page X): [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

Explanation:
A. [Correct/Incorrect]: [Self-contained explanation about only this option]
B. [Correct/Incorrect]: [Self-contained explanation about only this option]
C. [Correct/Incorrect]: [Self-contained explanation about only this option]
D. [Correct/Incorrect]: [Self-contained explanation about only this option]

Question 2 (Page X): [Question text]
...
```

After all questions, provide a brief 2-3 sentence summary of key concepts covered.
"""
        
        # Construct message content with both text and images
        message_content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add images for each page
        for page_data in chunk:
            message_content.append({
                "type": "image", 
                "source": {
                    "type": "base64", 
                    "media_type": "image/jpeg", 
                    "data": page_data["image_data"]
                }
            })
            
        return prompt, message_content

    def send_request_to_claude(self, message_content: List[Dict], max_tokens: int = 4000) -> Dict:
        """
        Send a request to Claude and handle rate limiting.
        
        Args:
            message_content: Message content to send to Claude
            max_tokens: Maximum tokens in response
            
        Returns:
            Response from Claude with token usage
        """
        max_retries = 5
        retry_count = 0
        retry_delay = 5  # initial delay in seconds
        
        while retry_count < max_retries:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": message_content}
                    ]
                )
                
                # Track token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                print(f"Response received - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                print(f"Running total - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}, Total: {self.total_input_tokens + self.total_output_tokens}")
                
                return {
                    "response": response.content[0].text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
            except anthropic.RateLimitError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise e
                
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
                
            except Exception as e:
                print(f"Error sending request to Claude: {e}")
                raise e

    def validate_and_correct_questions(self, questions_text: str, chunk: List[Dict], 
                                     retry_count: int = 0, max_retries: int = 2) -> Dict:
        """
        Validate questions and request corrections if needed.
        
        Args:
            questions_text: Text containing the questions to validate
            chunk: Current chunk of PDF pages (for correction if needed)
            retry_count: Current retry attempt
            max_retries: Maximum number of correction attempts
            
        Returns:
            Dict with validated/corrected questions and validation status
        """
        # Store original questions for first validation
        original_questions = questions_text if retry_count == 0 else None
        
        # Validate the questions
        print("\n" + "="*50)
        print(f"VALIDATION ATTEMPT #{retry_count + 1}")
        print("="*50)
        print("Validating questions...")
        validation_results = self.validator.validate_questions(questions_text)
        
        # Print validation summary
        print(f"Validation results: {validation_results['questions_with_issues']} of {validation_results['total_questions']} questions have issues")
        
        # Display specific issues
        if validation_results["issues"]:
            print("\nIssues found:")
            for issue in validation_results["issues"][:5]:  # Show first 5 issues
                print(f"- {issue}")
            if len(validation_results["issues"]) > 5:
                print(f"- ... and {len(validation_results['issues']) - 5} more issues")
        
        # If no issues or max retries reached, return the questions as is
        if not validation_results["issues"] or retry_count >= max_retries:
            if not validation_results["issues"]:
                print("\nAll questions passed validation!")
            elif retry_count >= max_retries:
                print(f"\nMax correction attempts ({max_retries}) reached. Using current version.")
                print(f"Remaining issues: {len(validation_results['issues'])}")
            
            return {
                "original_questions": original_questions,
                "questions_text": questions_text,
                "validation_results": validation_results,
                "is_valid": len(validation_results["issues"]) == 0,
                "correction_history": [{
                    "attempt": retry_count + 1,
                    "issues_count": len(validation_results["issues"]),
                    "issues": validation_results["issues"]
                }]
            }
        
        # Generate correction prompt
        print(f"\nFound {len(validation_results['issues'])} issues. Requesting corrections from Claude...")
        correction_prompt = self.validator.generate_correction_prompt(validation_results, questions_text)
        
        try:
            # Send correction request to Claude
            message_content = [{"type": "text", "text": correction_prompt}]
            correction_response = self.send_request_to_claude(message_content)
            corrected_questions = correction_response["response"]
            
            print(f"Received corrections. Validating corrected questions...")
            
            # Recursively validate the corrected questions
            next_result = self.validate_and_correct_questions(
                corrected_questions, 
                chunk,
                retry_count + 1, 
                max_retries
            )
            
            # Add current attempt to correction history
            if "correction_history" not in next_result:
                next_result["correction_history"] = []
                
            next_result["correction_history"].append({
                "attempt": retry_count + 1,
                "issues_count": len(validation_results["issues"]),
                "issues": validation_results["issues"]
            })
            
            # Ensure original questions are preserved
            if "original_questions" not in next_result and original_questions:
                next_result["original_questions"] = original_questions
                
            return next_result
            
        except Exception as e:
            print(f"Error requesting corrections: {e}")
            # Return original questions if correction fails
            return {
                "original_questions": original_questions,
                "questions_text": questions_text,
                "validation_results": validation_results,
                "is_valid": False,
                "error": str(e),
                "correction_history": [{
                    "attempt": retry_count + 1,
                    "issues_count": len(validation_results["issues"]),
                    "issues": validation_results["issues"]
                }]
            }

    def generate_questions_from_chunk(self, chunk: List[Dict], is_first_chunk: bool, 
                                      previous_summaries: Optional[str] = None, total_chunks: int = 1) -> Dict:
        """
        Generate MCQ questions from a PDF chunk using Claude Vision model.
        
        Args:
            chunk: Chunk of PDF pages as images
            is_first_chunk: Whether this is the first chunk
            previous_summaries: Combined summaries from previous chunks
            total_chunks: Total number of chunks in the document
            
        Returns:
            Dict containing generated questions and summary
        """
        if is_first_chunk:
            prompt, message_content = self.generate_first_chunk_prompt(chunk, total_chunks)
        else:
            prompt, message_content = self.generate_subsequent_chunk_prompt(chunk, previous_summaries)
        
        # Calculate estimated input tokens
        estimated_image_tokens = sum(page.get("estimated_tokens", 0) for page in chunk)
        estimated_text_tokens = len(prompt.split()) * 1.3  # Rough estimate
        estimated_total_tokens = estimated_image_tokens + estimated_text_tokens
            
        try:
            print(f"Sending request to Claude with estimated {int(estimated_total_tokens)} input tokens...")
            
            # Send request to Claude with images
            response = self.send_request_to_claude(message_content)
            questions_text = response["response"]
            
            # Validate and correct questions if needed
            validation_result = self.validate_and_correct_questions(questions_text, chunk)
            
            # Use the validated/corrected questions
            final_questions = validation_result["questions_text"]
            is_valid = validation_result["is_valid"]
            
            # Add validation status to the response
            response["response"] = final_questions
            response["validation_status"] = "Valid" if is_valid else "Issues Remain"
            response["validation_results"] = validation_result["validation_results"]
            
            return response
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Try with fewer images if there might be a size issue
            if len(chunk) > 5:
                print("Trying with reduced image quality or fewer images...")
                # Take every other page if there are many
                reduced_chunk = chunk[::2]
                return self.generate_questions_from_chunk(reduced_chunk, is_first_chunk, previous_summaries, total_chunks)
            return {
                "response": f"Error generating questions: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
                "validation_status": "Error"
            }

    def extract_summary_from_response(self, response: str) -> str:
        """
        Extract the summary from Claude's response.
        
        Args:
            response: Claude's full response
            
        Returns:
            Extracted summary
        """
        # Look for common patterns that might indicate a summary section
        summary_patterns = [
            r"Summary:(.*?)(?=\n\n|\Z)",
            r"Key concepts covered:(.*?)(?=\n\n|\Z)",
            r"Brief summary:(.*?)(?=\n\n|\Z)",
            r"Summary of key concepts:(.*?)(?=\n\n|\Z)",
            r"In summary,(.*?)(?=\n\n|\Z)"
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no clear summary section, check the last few paragraphs
        paragraphs = response.split("\n\n")
        for para in reversed(paragraphs):
            if len(para.split()) > 10 and "question" not in para.lower() and any(term in para.lower() for term in ["cover", "discuss", "introduce", "explore", "focus", "explain", "concept"]):
                return para.strip()
                
        # Fallback: take the last substantial paragraph
        for para in reversed(paragraphs):
            if len(para.split()) > 15 and not any(q in para.lower() for q in ["question", "option", "correct answer", "explanation"]):
                return para.strip()
                
        return "This section covers key educational concepts from the material."

    def process_pdf(self, file_path: str, output_dir: str = "quiz_output", target_pages_per_chunk: int = 30, 
                   skip_pages: int = 3, dpi: int = 300, optimize_tokens: bool = True):
        """
        Process the entire PDF, chunk it, generate and validate questions using vision models.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save output files
            target_pages_per_chunk: Target number of pages per chunk
            skip_pages: Number of initial pages to skip
            dpi: Resolution for rendering PDF pages
            optimize_tokens: Whether to optimize images for token usage
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract PDF pages as images with token optimization
        pages_data, effective_pages = self.extract_pages_as_images(file_path, skip_pages, dpi, optimize_tokens)
        if not pages_data:
            print("Error: Could not extract pages from PDF.")
            return
            
        print(f"Successfully extracted {effective_pages} pages from PDF as images.")
        
        # Chunk the pages
        chunks = self.chunk_pdf_pages(pages_data, effective_pages, target_pages_per_chunk)
        
        all_questions = []
        all_validation_results = []
        combined_summaries = ""
        token_usage = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"\n{'='*80}")
            print(f"Processing chunk {i+1}/{len(chunks)} (pages {chunk[0]['page_num']} to {chunk[-1]['page_num']})...")
            print(f"Chunk contains {len(chunk)} pages")
            print(f"{'='*80}")
            
            is_first_chunk = (i == 0)
            
            # Generate questions for this chunk with validation and correction
            result = self.generate_questions_from_chunk(
                chunk, 
                is_first_chunk, 
                combined_summaries if not is_first_chunk else None,
                total_chunks=len(chunks)
            )
            
            # Track token usage
            token_usage.append({
                "chunk": i+1,
                "pages": f"{chunk[0]['page_num']}-{chunk[-1]['page_num']}",
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
                "validation_status": result.get("validation_status", "Unknown")
            })
            
            # Store validation results
            if "validation_results" in result:
                all_validation_results.append({
                    "chunk": i+1,
                    "pages": f"{chunk[0]['page_num']}-{chunk[-1]['page_num']}",
                    "results": result["validation_results"]
                })
            
            # Extract and save summary for context in next chunks
            summary = self.extract_summary_from_response(result["response"])
            chunk_summary = f"Chunk {i+1} (pages {chunk[0]['page_num']}-{chunk[-1]['page_num']}): {summary}"
            self.summaries.append(chunk_summary)
            
            # Update combined summaries for next chunk
            if combined_summaries:
                combined_summaries += f"\n\n{chunk_summary}"
            else:
                combined_summaries = chunk_summary
                
            # Save the questions for this chunk
            all_questions.append(result["response"])
            
            # Save original and final questions for comparison
            has_original = "original_questions" in result and result["original_questions"] != result["response"]
            
            # Write individual chunk result to file
            with open(f"{output_dir}/chunk_{i+1}_questions.txt", "w", encoding="utf-8") as f:
                f.write(f"Questions for pages {chunk[0]['page_num']} to {chunk[-1]['page_num']}:\n\n")
                f.write(result["response"])
            
            # If there were corrections, save original questions and a comparison file
            if has_original:
                # Save original questions
                with open(f"{output_dir}/chunk_{i+1}_questions_original.txt", "w", encoding="utf-8") as f:
                    f.write(f"ORIGINAL Questions for pages {chunk[0]['page_num']} to {chunk[-1]['page_num']}:\n\n")
                    f.write(result["original_questions"])
                
                # Create comparison file showing both versions
                with open(f"{output_dir}/chunk_{i+1}_questions_comparison.txt", "w", encoding="utf-8") as f:
                    f.write(f"COMPARISON FOR CHUNK {i+1} (Pages {chunk[0]['page_num']}-{chunk[-1]['page_num']})\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write("ORIGINAL VERSION:\n")
                    f.write("-"*80 + "\n")
                    f.write(result["original_questions"])
                    f.write("\n\n")
                    
                    f.write("CORRECTED VERSION:\n")
                    f.write("-"*80 + "\n")
                    f.write(result["response"])
                    f.write("\n\n")
                    
                    # Add summary of corrections
                    if "correction_history" in result:
                        f.write("CORRECTION SUMMARY:\n")
                        f.write("-"*80 + "\n")
                        for correction in result["correction_history"]:
                            f.write(f"Attempt #{correction['attempt']}: Found {correction['issues_count']} issues\n")
                            if correction["issues"]:
                                f.write("Examples of issues:\n")
                                for issue in correction["issues"][:5]:  # Show first 5 issues
                                    f.write(f"- {issue}\n")
                                if len(correction["issues"]) > 5:
                                    f.write(f"- ... and {len(correction['issues']) - 5} more issues\n")
                            f.write("\n")
            
            # Write validation results to file
            if "validation_results" in result:
                with open(f"{output_dir}/chunk_{i+1}_validation.json", "w", encoding="utf-8") as f:
                    json.dump(result["validation_results"], f, indent=2)
                
                # Generate human-readable validation report
                suggestions = self.validator.suggest_corrections(result["validation_results"])
                with open(f"{output_dir}/chunk_{i+1}_validation_report.txt", "w", encoding="utf-8") as f:
                    f.write(f"VALIDATION REPORT FOR CHUNK {i+1} (Pages {chunk[0]['page_num']}-{chunk[-1]['page_num']})\n\n")
                    f.write(suggestions)
                
        # Write all questions to a single file
        with open(f"{output_dir}/all_questions.txt", "w", encoding="utf-8") as f:
            for i, questions in enumerate(all_questions):
                chunk = chunks[i]
                f.write(f"=== CHUNK {i+1} (PAGES {chunk[0]['page_num']} to {chunk[-1]['page_num']}) ===\n\n")
                f.write(questions)
                f.write("\n\n")
                
        # Write summaries to a file
        with open(f"{output_dir}/summaries.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.summaries))
            
        # Write token usage report
        with open(f"{output_dir}/token_usage.json", "w", encoding="utf-8") as f:
            json.dump({
                "chunks": token_usage,
                "total": {
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_input_tokens + self.total_output_tokens
                }
            }, f, indent=2)
            
        # Write validation results summary
        with open(f"{output_dir}/validation_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_validation_results, f, indent=2)
            
        # Create a human-readable token usage report
        with open(f"{output_dir}/token_usage_report.txt", "w", encoding="utf-8") as f:
            f.write("=== TOKEN USAGE REPORT ===\n\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"PDF: {os.path.basename(file_path)}\n")
            f.write(f"Pages processed: {effective_pages}\n\n")
            
            f.write("Per Chunk Breakdown:\n")
            for chunk_data in token_usage:
                f.write(f"  Chunk {chunk_data['chunk']} (Pages {chunk_data['pages']}):\n")
                f.write(f"    Input tokens:  {chunk_data['input_tokens']:,}\n")
                f.write(f"    Output tokens: {chunk_data['output_tokens']:,}\n")
                f.write(f"    Total tokens:  {chunk_data['total_tokens']:,}\n")
                f.write(f"    Validation:    {chunk_data['validation_status']}\n\n")
            
            f.write("Overall Usage:\n")
            f.write(f"  Total input tokens:  {self.total_input_tokens:,}\n")
            f.write(f"  Total output tokens: {self.total_output_tokens:,}\n")
            f.write(f"  Combined total:      {(self.total_input_tokens + self.total_output_tokens):,}\n\n")
            
            # Add cost estimate
            # Using approximate rates, adjust as needed
            input_cost_per_1m = 3.0  # $3.00 per 1M input tokens for Claude 3.7 Sonnet
            output_cost_per_1m = 15.0  # $15.00 per 1M output tokens for Claude 3.7 Sonnet
            
            input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_1m
            output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_1m
            total_cost = input_cost + output_cost
            
            f.write("Estimated Cost (USD):\n")
            f.write(f"  Input cost:  ${input_cost:.2f}\n")
            f.write(f"  Output cost: ${output_cost:.2f}\n")
            f.write(f"  Total cost:  ${total_cost:.2f}\n")
            
        print(f"\nProcessing complete. Results saved to {output_dir}/")
        print(f"Total token usage: {self.total_input_tokens:,} input + {self.total_output_tokens:,} output = {(self.total_input_tokens + self.total_output_tokens):,} tokens")
        print(f"Estimated cost: ${total_cost:.2f}")
        print(f"See {output_dir}/token_usage_report.txt for detailed breakdown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate quiz questions from PDF slides using Vision Models with automatic validation')
    parser.add_argument('--file', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', 
                        help='Claude model to use (must be vision-capable)')
    parser.add_argument('--target-pages', type=int, default=15, 
                       help='Target number of pages per chunk (default: 15)')
    parser.add_argument('--skip-pages', type=int, default=3,
                       help='Number of initial pages to skip (default: 3)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution for page rendering (default: 300)')
    parser.add_argument('--optimize-tokens', action='store_true', default=True,
                       help='Optimize images to reduce token usage')
    parser.add_argument('--high-quality', action='store_true',
                       help='Use higher quality images (more tokens)')
    parser.add_argument('--max-correction-attempts', type=int, default=2,
                       help='Maximum number of correction attempts per chunk (default: 2)')
    
    args = parser.parse_args()
    
    # If high-quality is specified, disable token optimization
    if args.high_quality:
        args.optimize_tokens = False
    
    # Validate model choice (must be vision-capable)
    valid_vision_models = [
        'claude-3-7-sonnet-20250219',
        'claude-3-opus-20240229', 
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307'
    ]
    
    if args.model not in valid_vision_models:
        print(f"Warning: The model {args.model} may not support vision capabilities.")
        print(f"Recommended models: {', '.join(valid_vision_models)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
            
    # Warn about token usage with high DPI
    if args.dpi > 150 and not args.optimize_tokens:
        pages_per_chunk = min(args.target_pages, 30)
        estimated_tokens = pages_per_chunk * 100000  # Very rough estimate
        estimated_cost = (estimated_tokens / 1_000_000) * 20
        
        print(f"WARNING: High resolution ({args.dpi} DPI) without token optimization")
        print(f"This could consume approximately {estimated_tokens:,} tokens per chunk")
        print(f"Estimated cost per chunk: ${estimated_cost:.2f}")
        response = input("Continue with these settings? (y/n): ")
        if response.lower() != 'y':
            print("Consider using --optimize-tokens or lowering --dpi to 150")
            exit()
    
    processor = PDFVisionProcessor(api_key=args.api_key, model=args.model)
    processor.process_pdf(file_path=args.file, 
                          output_dir=args.output_dir, 
                          target_pages_per_chunk=args.target_pages, 
                          skip_pages=args.skip_pages, 
                          dpi=args.dpi,
                          optimize_tokens=args.optimize_tokens)