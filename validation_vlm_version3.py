import os
import re
import json
import base64
import argparse
import fitz  # PyMuPDF
from PIL import Image
import io
import anthropic
import time
import math
import random
from typing import List, Dict, Tuple, Optional, Any

class MCQValidator:
    """
    A class to validate MCQ questions generated from PDFs using Claude.
    This script focuses on finding issues without making corrections.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the MCQ validator.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_request_time = 0
        
        # Rate limiting settings
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self.tokens_per_minute_limit = 100000  # Estimated rate limit
        self.base_sleep_time = 2.0  # Base sleep time in seconds
    
    def parse_questions_file(self, file_path: str) -> Dict:
        """
        Parse existing MCQ file into structured format.
        
        Args:
            file_path: Path to the MCQ file
            
        Returns:
            Dictionary with chunks and questions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks based on "=== CHUNK" headers
            chunk_pattern = r'===\s+CHUNK\s+(\d+)\s+\(PAGES\s+(\d+)\s+to\s+(\d+)\)\s+===\s*\n\n?((?:.|\n)*?)(?=\n*===|\Z)'
            chunk_matches = re.findall(chunk_pattern, content)
            
            chunks = []
            
            for chunk_idx, start_page, end_page, chunk_content in chunk_matches:
                # Clean up chunk content - remove markdown code blocks
                chunk_content = re.sub(r'```', '', chunk_content).strip()
                
                # Parse questions in this chunk
                questions = self._parse_questions_in_chunk(chunk_content, int(chunk_idx))
                
                chunks.append({
                    'chunk_idx': int(chunk_idx),
                    'start_page': int(start_page),
                    'end_page': int(end_page),
                    'questions': questions,
                    'original_content': chunk_content
                })
            
            # If no chunks were found, try to parse the entire content as a single chunk
            if not chunks:
                # Remove markdown code blocks
                content = re.sub(r'```', '', content).strip()
                questions = self._parse_questions_in_chunk(content, 1)
                
                if questions:
                    # Try to find page range in the content
                    page_range_match = re.search(r'PAGES\s+(\d+)\s+to\s+(\d+)', content)
                    if page_range_match:
                        start_page, end_page = page_range_match.groups()
                    else:
                        start_page, end_page = 1, 100  # Default
                    
                    chunks.append({
                        'chunk_idx': 1,
                        'start_page': int(start_page),
                        'end_page': int(end_page),
                        'questions': questions,
                        'original_content': content
                    })
            
            return {
                'chunks': chunks,
                'total_questions': sum(len(chunk['questions']) for chunk in chunks)
            }
                
        except Exception as e:
            print(f"Error parsing questions file: {e}")
            return {'chunks': [], 'total_questions': 0}
    
    def _parse_questions_in_chunk(self, chunk_content: str, chunk_idx: int) -> List[Dict]:
        """
        Parse questions in a chunk.
        
        Args:
            chunk_content: Text content of the chunk
            chunk_idx: Index of the chunk
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        # Find all questions in the chunk
        question_pattern = r'Question\s+(\d+)\s+\(((?:Page|Pages)\s+[^)]+)\):\s*(.*?)(?=\n\nQuestion|\n\n===|\Z)'
        question_matches = re.findall(question_pattern, chunk_content, re.DOTALL)
        
        for q_num, page_ref, q_content in question_matches:
            try:
                # Parse question content (options and explanation)
                question_dict = self._parse_question_content(q_content.strip(), int(q_num), page_ref, chunk_idx)
                
                if question_dict:
                    questions.append(question_dict)
            except Exception as e:
                print(f"Error parsing Question {q_num}: {e}")
        
        return questions
    
    def _parse_question_content(self, content: str, q_num: int, page_ref: str, chunk_idx: int) -> Optional[Dict]:
        """
        Parse the content of a single question.
        
        Args:
            content: Question content text
            q_num: Question number
            page_ref: Page reference
            chunk_idx: Index of the chunk
            
        Returns:
            Structured question dictionary
        """
        try:
            # Split content into question text, options, and explanation
            parts = content.split('\n\nExplanation:')
            
            if len(parts) != 2:
                print(f"Warning: Question {q_num} does not have a clear explanation section.")
                return None
            
            options_part = parts[0].strip()
            explanation_part = parts[1].strip()
            
            # Extract options
            option_pattern = r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|\Z)'
            option_matches = re.findall(option_pattern, options_part, re.DOTALL)
            
            options = {}
            for opt_letter, opt_text in option_matches:
                options[opt_letter] = opt_text.strip()
            
            # Extract explanation for each option
            explanation_pattern = r'([A-D])\.\s*(.*?)(?=\n[A-D]\.|\Z)'
            explanation_matches = re.findall(explanation_pattern, explanation_part, re.DOTALL)
            
            explanations = {}
            for exp_letter, exp_text in explanation_matches:
                explanations[exp_letter] = exp_text.strip()
            
            # Extract the question text (everything before the first option)
            question_text_match = re.match(r'(.*?)(?=\nA\.)', options_part, re.DOTALL)
            question_text = question_text_match.group(1).strip() if question_text_match else ""
            
            # Determine correct options
            correct_options = []
            for option, explanation in explanations.items():
                if explanation.lower().startswith("correct:"):
                    correct_options.append(option)
            
            # Extract page numbers for reference
            page_numbers = []
            if "Pages" in page_ref:
                page_ref_clean = page_ref.replace("Pages", "").strip()
                if "-" in page_ref_clean:
                    start, end = map(int, page_ref_clean.split("-"))
                    page_numbers = list(range(start, end + 1))
                else:
                    try:
                        page_numbers = [int(page_ref_clean)]
                    except ValueError:
                        page_numbers = []
            elif "Page" in page_ref:
                page_ref_clean = page_ref.replace("Page", "").strip()
                try:
                    page_numbers = [int(page_ref_clean)]
                except ValueError:
                    page_numbers = []
            
            # Create the question dictionary
            return {
                'question_num': q_num,
                'page_ref': page_ref.strip(),
                'page_numbers': page_numbers,
                'question_text': question_text,
                'options': options,
                'explanations': explanations,
                'correct_options': correct_options,
                'chunk_idx': chunk_idx,
                'original_content': content
            }
        
        except Exception as e:
            print(f"Error parsing content for Question {q_num}: {e}")
            return None

    def extract_pdf_chunk_as_images(self, pdf_path: str, start_page: int, end_page: int, dpi: int = 300) -> List[Dict]:
        """
        Extract a chunk of pages from a PDF as base64 encoded images.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Start page (1-indexed)
            end_page: End page (1-indexed)
            dpi: Resolution for page rendering
            
        Returns:
            List of dictionaries with page images
        """
        page_images = []
        
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            # Adjust page range to be within the document
            start_page = max(1, start_page)
            end_page = min(total_pages, end_page)
            
            # Convert to 0-indexed
            start_idx = start_page - 1
            end_idx = end_page - 1
            
            print(f"Extracting pages {start_page} to {end_page} from PDF...")
            
            # Process each page
            for page_idx in range(start_idx, end_idx + 1):
                # Get the page
                page = pdf_document.load_page(page_idx)
                
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                img_data = pix.tobytes()
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB mode if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if larger than 1000 pixels in any dimension to reduce token usage
                max_dimension = 1000
                if img.width > max_dimension or img.height > max_dimension:
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=75)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Add to list
                page_images.append({
                    'page_num': page_idx + 1,  # 1-indexed
                    'image_data': img_base64
                })
            
            pdf_document.close()
            return page_images
            
        except Exception as e:
            print(f"Error extracting pages as images: {e}")
            return []

    def generate_validation_prompt(self, chunk: Dict) -> str:
        """
        Generate a prompt for Claude to only validate (not correct) questions in a chunk.
        
        Args:
            chunk: Chunk dictionary with questions
            
        Returns:
            Validation prompt
        """
        # Format all questions in the chunk
        questions_text = []
        
        for question in chunk['questions']:
            q_text = f"Question {question['question_num']} ({question['page_ref']}): {question['question_text']}"
            
            for option in ['A', 'B', 'C', 'D']:
                if option in question['options']:
                    q_text += f"\n{option}. {question['options'][option]}"
            
            q_text += "\n\nExplanation:"
            
            for option in ['A', 'B', 'C', 'D']:
                if option in question['explanations']:
                    q_text += f"\n{option}. {question['explanations'][option]}"
            
            questions_text.append(q_text)
        
        combined_questions = "\n\n".join(questions_text)
        
        prompt = f"""You are a subject matter expert VALIDATING multiple-choice questions from a tax document. ONLY IDENTIFY ISSUES in the questions without rewriting them. Focus on finding problems that need correction.

QUESTIONS TO VALIDATE:

{combined_questions}

VALIDATION CRITERIA:
1. Ensure exactly ONE option is correct for each question
2. CRITICAL: Explanations for INCORRECT options must stand completely on their own and NEVER hint at or reference the correct answer
3. Abbreviations must be expanded at least once (e.g., "Internal Revenue Service (IRS)")
4. Questions must be clear and specific
5. All explanations must start with "Correct:" or "Incorrect:"
6. DO NOT use phrases like "according to the source material" in explanations
7. Present information as established facts in the field

CRITICAL ISSUES TO IDENTIFY IN INCORRECT EXPLANATIONS:
- References to the correct answer or option
- Comparative phrases like "rather than", "instead of", "unlike", etc.
- Citations to "source material" or similar phrases
- Any hints about which option is correct

FORBIDDEN PHRASES TO IDENTIFY IN EXPLANATIONS:
- "according to the source material"
- "as explicitly stated in the source material"
- "the source material on page"
- "in the source material"
- "unlike the correct answer"
- "the correct answer is"
- "this confuses X with Y"
- "provided by the correct answer"
- "this is not what the question is asking for"
- "this does not address the question"
- "this would be correct if"
- "unlike other options"
- "compared to"
- "rather than"
- "instead of"
- "As presented in"
- "As shown in"
- "As mentioned in"

IMPORTANT INSTRUCTIONS:
1. DO NOT rewrite or correct the questions - ONLY identify issues
2. Based on the PDF content provided as images, list all issues for each question
3. Be specific about which option or explanation has issues
4. Specify if any question has multiple correct answers

YOUR RESPONSE FORMAT:
```
QUESTION 1 (Page X): VALIDATION ISSUES
- [List specific issues with this question]
- [Be specific about which option or explanation has the problem]
- [Include quotes from the problematic parts]

QUESTION 2 (Page Y): VALIDATION ISSUES
- [List specific issues with this question]
...
```

If a question has no issues, you can indicate "No issues detected."
DO NOT REWRITE THE QUESTIONS OR PROVIDE CORRECTIONS - JUST IDENTIFY THE ISSUES.
"""
        
        return prompt

    def parse_validation_response(self, response_text: str, original_chunk: Dict) -> Dict:
        """
        Parse Claude's validation response to extract identified issues.
        
        Args:
            response_text: Claude's response text
            original_chunk: Original chunk dictionary
            
        Returns:
            Dictionary with validation results
        """
        validation_chunk = {
            'chunk_idx': original_chunk['chunk_idx'],
            'start_page': original_chunk['start_page'],
            'end_page': original_chunk['end_page'],
            'questions': []
        }
        
        # Clean the response text
        clean_response = re.sub(r'```', '', response_text).strip()
        
        # Find all question validations in the response
        question_pattern = r'QUESTION\s+(\d+)\s+\([^)]+\):\s+VALIDATION\s+ISSUES\s*(.*?)(?=\n\nQUESTION|\Z)'
        question_matches = re.findall(question_pattern, clean_response, re.DOTALL)
        
        for q_num_str, validation_content in question_matches:
            try:
                q_num = int(q_num_str)
                
                # Find the original question
                original_question = next((q for q in original_chunk['questions'] if q['question_num'] == q_num), None)
                
                if not original_question:
                    print(f"Warning: Could not find original Question {q_num} in chunk {original_chunk['chunk_idx']}.")
                    continue
                
                # Determine if the question has issues
                has_issues = "no issues detected" not in validation_content.lower()
                
                # Extract issues by category
                issues = {
                    'has_issues': has_issues,
                    'cross_references': [],
                    'forbidden_phrases': [],
                    'multiple_correct_answers': "multiple correct" in validation_content.lower(),
                    'no_correct_answers': "no correct" in validation_content.lower(),
                    'missing_abbreviation_expansion': [],
                    'unclear_question': "unclear" in validation_content.lower() or "ambiguous" in validation_content.lower(),
                    'raw_issues': validation_content.strip()
                }
                
                # Look for abbreviations that need expansion
                abbr_pattern = r'missing abbreviation.*?([A-Z]{2,})'
                abbr_matches = re.findall(abbr_pattern, validation_content, re.IGNORECASE)
                issues['missing_abbreviation_expansion'] = list(set(abbr_matches))
                
                # Look for cross-references
                for opt in ['A', 'B', 'C', 'D']:
                    if f"Option {opt}" in validation_content and "reference" in validation_content.lower():
                        # Simple heuristic to extract which option is being referenced
                        for correct_opt in original_question['correct_options']:
                            if correct_opt in validation_content and correct_opt != opt:
                                issues['cross_references'].append({
                                    'option': opt,
                                    'reference': correct_opt,
                                    'explanation': original_question['explanations'].get(opt, "")
                                })
                
                # Look for forbidden phrases
                forbidden_phrases = [
                    "according to the source material", "in the source material", 
                    "rather than", "instead of", "unlike", "compared to", 
                    "correct answer", "would be correct if"
                ]
                
                for phrase in forbidden_phrases:
                    if phrase.lower() in validation_content.lower():
                        for opt in ['A', 'B', 'C', 'D']:
                            if f"Option {opt}" in validation_content and phrase.lower() in validation_content.lower():
                                issues['forbidden_phrases'].append({
                                    'option': opt,
                                    'phrase': phrase,
                                    'explanation': original_question['explanations'].get(opt, "")
                                })
                
                validation_chunk['questions'].append({
                    'question_num': q_num,
                    'page_ref': original_question['page_ref'],
                    'issues': issues,
                    'original_question': original_question
                })
            
            except Exception as e:
                print(f"Error parsing validation for Question {q_num_str}: {e}")
        
        # If any questions are missing in the validation set, assume they have no issues
        for original_question in original_chunk['questions']:
            if not any(q['question_num'] == original_question['question_num'] for q in validation_chunk['questions']):
                validation_chunk['questions'].append({
                    'question_num': original_question['question_num'],
                    'page_ref': original_question['page_ref'],
                    'issues': {
                        'has_issues': False,
                        'cross_references': [],
                        'forbidden_phrases': [],
                        'multiple_correct_answers': False,
                        'no_correct_answers': False,
                        'missing_abbreviation_expansion': [],
                        'unclear_question': False,
                        'raw_issues': "No issues detected."
                    },
                    'original_question': original_question
                })
        
        # Sort questions by question number
        validation_chunk['questions'].sort(key=lambda q: q['question_num'])
        
        return validation_chunk

    def calculate_sleep_time(self, tokens_used: int) -> float:
        """
        Calculate sleep time based on token usage to avoid rate limiting.
        
        Args:
            tokens_used: Number of tokens used in the last request
            
        Returns:
            Sleep time in seconds
        """
        # Calculate minimum time needed based on token rate limit
        min_sleep_time = (tokens_used / self.tokens_per_minute_limit) * 60.0
        
        # Add jitter to avoid synchronized requests
        jitter = random.uniform(0, 1.0)
        
        # Calculate final sleep time
        sleep_time = max(self.min_request_interval, self.base_sleep_time, min_sleep_time) + jitter
        
        return sleep_time

    def validate_chunk(self, chunk: Dict, pdf_path: str) -> Dict:
        """
        Validate all questions in a chunk using Claude with rate limiting.
        
        Args:
            chunk: Chunk dictionary with questions
            pdf_path: Path to the PDF file
            
        Returns:
            Validation result dictionary
        """
        # Extract chunk pages as images
        pdf_pages = self.extract_pdf_chunk_as_images(pdf_path, chunk['start_page'], chunk['end_page'])
        
        if not pdf_pages:
            print(f"Warning: Could not extract pages for chunk {chunk['chunk_idx']}.")
            # Return basic validation with no issues
            return {
                'chunk_idx': chunk['chunk_idx'],
                'start_page': chunk['start_page'],
                'end_page': chunk['end_page'],
                'questions': [{
                    'question_num': q['question_num'],
                    'page_ref': q['page_ref'],
                    'issues': {
                        'has_issues': False,
                        'raw_issues': "Could not validate due to PDF extraction failure."
                    },
                    'original_question': q
                } for q in chunk['questions']]
            }
        
        # Generate validation prompt
        prompt = self.generate_validation_prompt(chunk)
        
        # Apply rate limiting - wait if needed
        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        
        # If less than minimum interval has passed, sleep
        if elapsed_time < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed_time + random.uniform(0, 0.5)
            print(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        try:
            # Prepare message content with text and images
            message_content = [{"type": "text", "text": prompt}]
            
            # Add images of relevant pages (limit to 10 to avoid token issues)
            for page_data in pdf_pages[:10]:
                message_content.append({
                    "type": "image", 
                    "source": {
                        "type": "base64", 
                        "media_type": "image/jpeg", 
                        "data": page_data['image_data']
                    }
                })
            
            print(f"Validating Chunk {chunk['chunk_idx']} with {len(chunk['questions'])} questions...")
            
            # Record request time
            self.last_request_time = time.time()
            
            # Send request to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": message_content}
                ]
            )
            
            # Track token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            print(f"  - Used {input_tokens:,} input + {output_tokens:,} output = {total_tokens:,} total tokens")
            
            # Calculate and apply dynamic sleep based on token usage
            sleep_time = self.calculate_sleep_time(total_tokens)
            print(f"  - Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
            # Parse validation response
            validation_chunk = self.parse_validation_response(response.content[0].text, chunk)
            
            print(f"Completed validation of Chunk {chunk['chunk_idx']}")
            return validation_chunk
            
        except Exception as e:
            print(f"Error validating chunk {chunk['chunk_idx']}: {e}")
            
            # Sleep longer on error to recover from rate limits
            print(f"Sleeping for 60 seconds after error...")
            time.sleep(60)
            
            # Return basic validation with no issues on error
            return {
                'chunk_idx': chunk['chunk_idx'],
                'start_page': chunk['start_page'],
                'end_page': chunk['end_page'],
                'questions': [{
                    'question_num': q['question_num'],
                    'page_ref': q['page_ref'],
                    'issues': {
                        'has_issues': False,
                        'raw_issues': f"Could not validate due to error: {str(e)}"
                    },
                    'original_question': q
                } for q in chunk['questions']]
            }

    def generate_validation_report(self, validation_results: Dict, output_dir: str):
        """
        Generate a detailed report of validation results.
        
        Args:
            validation_results: Validation results
            output_dir: Directory to save the report
        """
        # Count the number of questions with issues
        questions_with_issues = 0
        for chunk in validation_results['chunks']:
            for question in chunk['questions']:
                if question['issues']['has_issues']:
                    questions_with_issues += 1
        
        report = [
            "# MCQ Validation Report",
            "",
            "## Summary",
            f"- Total questions: {validation_results['total_questions']}",
            f"- Questions with issues: {questions_with_issues}",
            f"- Percentage with issues: {(questions_with_issues / validation_results['total_questions'] * 100):.1f}%",
            "",
            "## Token Usage",
            f"- Input tokens: {self.total_input_tokens:,}",
            f"- Output tokens: {self.total_output_tokens:,}",
            f"- Total tokens: {(self.total_input_tokens + self.total_output_tokens):,}",
            "",
            "## Validation by Chunk"
        ]
        
        # Add details for each chunk
        for chunk in validation_results['chunks']:
            chunk_issues = sum(1 for q in chunk['questions'] if q['issues']['has_issues'])
            
            report.append(f"### Chunk {chunk['chunk_idx']} (Pages {chunk['start_page']} to {chunk['end_page']})")
            report.append(f"- Questions in chunk: {len(chunk['questions'])}")
            report.append(f"- Questions with issues: {chunk_issues}")
            report.append("")
            
            # Add details for each question with issues
            for question in chunk['questions']:
                if question['issues']['has_issues']:
                    report.append(f"#### Question {question['question_num']} ({question['page_ref']})")
                    report.append("")
                    report.append("**Issues detected:**")
                    
                    # Format the raw issues with proper markdown
                    raw_issues = question['issues']['raw_issues'].strip()
                    formatted_issues = []
                    
                    for line in raw_issues.split('\n'):
                        if line.strip():
                            if line.strip().startswith('-'):
                                formatted_issues.append(line)
                            else:
                                formatted_issues.append(f"- {line}")
                    
                    report.append("\n".join(formatted_issues))
                    report.append("")
        
        # Write report to file
        report_path = os.path.join(output_dir, "validation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"Validation report saved to {report_path}")

    def save_validation_results(self, validation_results: Dict, output_dir: str):
        """
        Save validation results to a JSON file.
        
        Args:
            validation_results: Validation results
            output_dir: Directory to save the results
        """
        json_path = os.path.join(output_dir, "validation_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
            
        print(f"Validation results saved to {json_path}")

    def validate_mcqs(self, question_file: str, pdf_path: str, output_dir: str = "validation_output"):
        """
        Main method to validate MCQs using Claude.
        
        Args:
            question_file: Path to the questions file
            pdf_path: Path to the source PDF
            output_dir: Directory to save output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Reading questions from {question_file}")
        parsed_data = self.parse_questions_file(question_file)
        
        if parsed_data['total_questions'] == 0:
            print("Error: No questions found in the input file.")
            return
            
        print(f"Found {parsed_data['total_questions']} questions in {len(parsed_data['chunks'])} chunks")
        
        # Validate each chunk
        validation_results = {
            'chunks': [],
            'total_questions': parsed_data['total_questions']
        }
        
        for chunk in parsed_data['chunks']:
            validation_chunk = self.validate_chunk(chunk, pdf_path)
            validation_results['chunks'].append(validation_chunk)
        
        # Generate validation report
        print("Generating validation report...")
        self.generate_validation_report(validation_results, output_dir)
        
        # Save validation results
        print("Saving validation results...")
        self.save_validation_results(validation_results, output_dir)
        
        print(f"Validation complete. Results saved to {output_dir}/")
        print(f"Token usage: {self.total_input_tokens:,} input + {self.total_output_tokens:,} output = {(self.total_input_tokens + self.total_output_tokens):,} tokens")

def main():
    parser = argparse.ArgumentParser(description='Validate MCQ questions using Claude')
    parser.add_argument('--questions', type=str, required=True, help='Path to questions file')
    parser.add_argument('--pdf', type=str, required=True, help='Path to source PDF file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='validation_output', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', 
                      help='Claude model to use (must be vision-capable)')
    
    args = parser.parse_args()
    
    # Initialize validator and run validation
    validator = MCQValidator(api_key=args.api_key, model=args.model)
    validator.validate_mcqs(
        question_file=args.questions,
        pdf_path=args.pdf,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()