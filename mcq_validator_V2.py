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
    A class to validate and fix MCQ questions generated from PDFs.
    Uses Claude to perform validation and correction while maintaining the original format.
    Processes all questions in a chunk together with rate limiting.
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
        Generate a prompt for Claude to validate all questions in a chunk.
        
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
        
        prompt = f"""You are a subject matter expert CORRECTING multiple-choice questions from a tax document. You must ACTIVELY REWRITE any problematic questions and explanations following these criteria.

QUESTIONS TO CORRECT:

{combined_questions}

CRITICAL CORRECTION PRIORITIES:
1. MOST IMPORTANT: COMPLETELY REWRITE any explanation for incorrect options that references the correct answer
2. Ensure exactly ONE option is correct for each question
3. REWRITE all explanations to be completely self-contained
4. Fix any unclear or ambiguous question text
5. Expand all abbreviations at least once (e.g., "Internal Revenue Service (IRS)")

FOR INCORRECT OPTION EXPLANATIONS:
- ALWAYS completely rewrite any explanation that mentions, hints at, or references the correct answer
- Each explanation must stand COMPLETELY on its own as if you don't know the other options
- Never use comparative language ("rather than," "instead of," "unlike")
- Never indicate which option is correct
- NEVER use phrases like "according to the source material" or "as stated in the text"
- Present information as established facts, not as citations of the material

WHEN MULTIPLE CORRECT ANSWERS EXIST:
- Add an "All of the above" option, OR
- Replace options with "Both A and B are correct," OR
- Modify the question to make only one answer correct

FORBIDDEN PHRASES IN EXPLANATIONS:
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

YOUR RESPONSE MUST BE THE FULLY CORRECTED QUESTIONS IN THIS FORMAT:
```
Question 1 (Page X): [Corrected question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

Explanation:
A. [Correct/Incorrect]: [Self-contained explanation]
B. [Correct/Incorrect]: [Self-contained explanation]
C. [Correct/Incorrect]: [Self-contained explanation]
D. [Correct/Incorrect]: [Self-contained explanation]

Question 2 (Page Y): [Corrected question text]
...
```

Follow this format EXACTLY, with one blank line between the options and "Explanation:" and two blank lines between questions.

I will reject any response that doesn't provide FULLY CORRECTED questions with self-contained explanations.
"""
        
        return prompt

    def parse_validation_response(self, response_text: str, original_chunk: Dict) -> Dict:
        """
        Parse Claude's validation response to extract corrected questions.
        
        Args:
            response_text: Claude's response text
            original_chunk: Original chunk dictionary
            
        Returns:
            Dictionary with corrected questions
        """
        # Extract the corrected questions section
        corrected_chunk = original_chunk.copy()
        corrected_questions = []
        
        # Remove markdown code blocks if present
        clean_response = re.sub(r'```', '', response_text).strip()
        
        # Find all corrected questions in the response
        question_pattern = r'Question\s+(\d+)\s+\(((?:Page|Pages)\s+[^)]+)\):\s*(.*?)(?=\n\nQuestion|\Z)'
        question_matches = re.findall(question_pattern, clean_response, re.DOTALL)
        
        for q_num_str, page_ref, q_content in question_matches:
            try:
                q_num = int(q_num_str)
                
                # Find the original question
                original_question = next((q for q in original_chunk['questions'] if q['question_num'] == q_num), None)
                
                if not original_question:
                    print(f"Warning: Could not find original Question {q_num} in chunk {original_chunk['chunk_idx']}.")
                    continue
                
                # Parse the corrected question content
                corrected_question = self._parse_question_content(
                    q_content.strip(), q_num, page_ref, original_chunk['chunk_idx']
                )
                
                if corrected_question:
                    # Mark as corrected and keep original page numbers
                    corrected_question['was_corrected'] = True
                    corrected_question['page_numbers'] = original_question['page_numbers']
                    corrected_questions.append(corrected_question)
                else:
                    print(f"Warning: Could not parse corrected Question {q_num}.")
                    original_question['was_corrected'] = False
                    corrected_questions.append(original_question)
            
            except Exception as e:
                print(f"Error parsing corrected Question {q_num_str}: {e}")
        
        # If any questions are missing in the corrected set, use the originals
        for original_question in original_chunk['questions']:
            if not any(q['question_num'] == original_question['question_num'] for q in corrected_questions):
                original_question['was_corrected'] = False
                corrected_questions.append(original_question)
        
        # Sort questions by question number
        corrected_questions.sort(key=lambda q: q['question_num'])
        
        # Update the chunk with corrected questions
        corrected_chunk['questions'] = corrected_questions
        return corrected_chunk

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

    def format_questions_for_output(self, chunk: Dict) -> str:
        """
        Format questions in a chunk for output.
        
        Args:
            chunk: Chunk dictionary with questions
            
        Returns:
            Formatted questions text
        """
        output = [f"=== CHUNK {chunk['chunk_idx']} (PAGES {chunk['start_page']} to {chunk['end_page']}) ===\n"]
        
        # Add each question
        for i, question in enumerate(chunk['questions']):
            q_text = f"Question {question['question_num']} ({question['page_ref']}): {question['question_text']}"
            
            # Add options
            for option in ['A', 'B', 'C', 'D']:
                if option in question['options']:
                    q_text += f"\n{option}. {question['options'][option]}"
            
            # Add explanation section
            q_text += "\n\nExplanation:"
            
            for option in ['A', 'B', 'C', 'D']:
                if option in question['explanations']:
                    q_text += f"\n{option}. {question['explanations'][option]}"
            
            output.append(q_text)
        
        return "\n\n".join(output)

    def check_explanations_for_cross_references(self, chunk: Dict) -> Dict:
        """
        Check if explanations for incorrect options reference the correct answer.
        
        Args:
            chunk: Chunk with corrected questions
            
        Returns:
            Chunk with any issues fixed
        """
        for question in chunk['questions']:
            # Find the correct option(s)
            correct_options = []
            for option, explanation in question['explanations'].items():
                if explanation.lower().startswith("correct:"):
                    correct_options.append(option)
            
            # Skip if no clear correct option is found
            if not correct_options:
                continue
            
            # For each incorrect option, check if it references the correct ones
            issues_found = False
            for option, explanation in question['explanations'].items():
                if explanation.lower().startswith("incorrect:"):
                    # Check for mentions of the correct options
                    for correct_opt in correct_options:
                        if correct_opt in explanation:
                            print(f"Warning: Explanation for option {option} in Question {question['question_num']} references correct option {correct_opt}")
                            issues_found = True
                    
                    # Check for common comparative phrases
                    comparative_phrases = ["rather than", "instead of", "unlike", "compared to", "correct answer"]
                    for phrase in comparative_phrases:
                        if phrase.lower() in explanation.lower():
                            print(f"Warning: Explanation for option {option} in Question {question['question_num']} uses comparative phrase '{phrase}'")
                            issues_found = True
            
            if issues_found:
                question['needs_more_fixing'] = True
        
        return chunk

    def validate_chunk(self, chunk: Dict, pdf_path: str) -> Dict:
        """
        Validate all questions in a chunk using Claude with rate limiting.
        
        Args:
            chunk: Chunk dictionary with questions
            pdf_path: Path to the PDF file
            
        Returns:
            Corrected chunk dictionary
        """
        # Extract chunk pages as images
        pdf_pages = self.extract_pdf_chunk_as_images(pdf_path, chunk['start_page'], chunk['end_page'])
        
        if not pdf_pages:
            print(f"Warning: Could not extract pages for chunk {chunk['chunk_idx']}.")
            return chunk
        
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
            corrected_chunk = self.parse_validation_response(response.content[0].text, chunk)
            
            # Check corrected explanations for any remaining cross-references
            corrected_chunk = self.check_explanations_for_cross_references(corrected_chunk)
            
            print(f"Completed validation of Chunk {chunk['chunk_idx']}")
            return corrected_chunk
            
        except Exception as e:
            print(f"Error validating chunk {chunk['chunk_idx']}: {e}")
            
            # Sleep longer on error to recover from rate limits
            print(f"Sleeping for 60 seconds after error...")
            time.sleep(60)
            
            return chunk

    def generate_validation_report(self, original_data: Dict, corrected_data: Dict, output_dir: str):
        """
        Generate a detailed report of validation results.
        
        Args:
            original_data: Original parsed questions data
            corrected_data: Corrected questions data
            output_dir: Directory to save the report
        """
        # Count the number of questions corrected
        corrected_count = 0
        for chunk in corrected_data['chunks']:
            for question in chunk['questions']:
                if question.get('was_corrected', False):
                    corrected_count += 1
        
        # Count questions that need more fixing
        more_fixing_needed = 0
        for chunk in corrected_data['chunks']:
            for question in chunk['questions']:
                if question.get('needs_more_fixing', False):
                    more_fixing_needed += 1
        
        report = [
            "# MCQ Validation Report",
            "",
            "## Summary",
            f"- Total questions: {corrected_data['total_questions']}",
            f"- Questions corrected: {corrected_count}",
            f"- Percentage corrected: {(corrected_count / corrected_data['total_questions'] * 100):.1f}%",
            f"- Questions needing additional review: {more_fixing_needed}",
            "",
            "## Token Usage",
            f"- Input tokens: {self.total_input_tokens:,}",
            f"- Output tokens: {self.total_output_tokens:,}",
            f"- Total tokens: {(self.total_input_tokens + self.total_output_tokens):,}",
            "",
            "## Validation by Chunk"
        ]
        
        # Add details for each chunk
        for chunk in corrected_data['chunks']:
            chunk_corrected = sum(1 for q in chunk['questions'] if q.get('was_corrected', False))
            chunk_needs_fixing = sum(1 for q in chunk['questions'] if q.get('needs_more_fixing', False))
            
            report.append(f"### Chunk {chunk['chunk_idx']} (Pages {chunk['start_page']} to {chunk['end_page']})")
            report.append(f"- Questions in chunk: {len(chunk['questions'])}")
            report.append(f"- Questions corrected: {chunk_corrected}")
            if chunk_needs_fixing > 0:
                report.append(f"- Questions needing additional review: {chunk_needs_fixing}")
            report.append("")
        
        # Add list of questions needing review
        if more_fixing_needed > 0:
            report.append("## Questions Needing Additional Review")
            report.append("")
            report.append("The following questions may still have explanations that reference the correct answer:")
            report.append("")
            
            for chunk in corrected_data['chunks']:
                for question in chunk['questions']:
                    if question.get('needs_more_fixing', False):
                        report.append(f"- Question {question['question_num']} in Chunk {chunk['chunk_idx']}")
            
            report.append("")
        
        # Write report to file
        report_path = os.path.join(output_dir, "validation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"Validation report saved to {report_path}")

    def save_corrected_questions(self, corrected_data: Dict, output_dir: str):
        """
        Save corrected questions to file.
        
        Args:
            corrected_data: Corrected questions data
            output_dir: Directory to save the output
        """
        # Format all chunks
        formatted_chunks = []
        
        for chunk in corrected_data['chunks']:
            formatted_chunks.append(self.format_questions_for_output(chunk))
        
        # Combine all chunks
        all_questions_text = "\n\n".join(formatted_chunks)
        
        # Write to file
        output_path = os.path.join(output_dir, "corrected_questions.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_questions_text)
            
        print(f"Corrected questions saved to {output_path}")
        
        # Also save structured data in JSON format
        json_data = {
            'chunks': []
        }
        
        for chunk in corrected_data['chunks']:
            json_chunk = {
                'chunk_idx': chunk['chunk_idx'],
                'start_page': chunk['start_page'],
                'end_page': chunk['end_page'],
                'questions': []
            }
            
            for question in chunk['questions']:
                json_question = {
                    'question_num': question['question_num'],
                    'page_ref': question['page_ref'],
                    'question_text': question['question_text'],
                    'options': question['options'],
                    'explanations': question['explanations'],
                    'correct_options': question.get('correct_options', []),
                    'was_corrected': question.get('was_corrected', False),
                    'needs_more_fixing': question.get('needs_more_fixing', False)
                }
                
                json_chunk['questions'].append(json_question)
            
            json_data['chunks'].append(json_chunk)
        
        json_path = os.path.join(output_dir, "corrected_questions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        print(f"Structured data saved to {json_path}")

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
        corrected_data = {
            'chunks': [],
            'total_questions': parsed_data['total_questions']
        }
        
        for chunk in parsed_data['chunks']:
            corrected_chunk = self.validate_chunk(chunk, pdf_path)
            corrected_data['chunks'].append(corrected_chunk)
        
        # Generate validation report
        print("Generating validation report...")
        self.generate_validation_report(parsed_data, corrected_data, output_dir)
        
        # Save corrected questions
        print("Saving corrected questions...")
        self.save_corrected_questions(corrected_data, output_dir)
        
        print(f"Validation complete. Results saved to {output_dir}/")
        print(f"Token usage: {self.total_input_tokens:,} input + {self.total_output_tokens:,} output = {(self.total_input_tokens + self.total_output_tokens):,} tokens")

def main():
    parser = argparse.ArgumentParser(description='Validate MCQ questions from PDFs using Claude')
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