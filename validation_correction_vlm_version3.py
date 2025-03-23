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

class MCQCorrector:
    """
    A class to correct MCQ questions using Claude based on validation results.
    Takes validation results and original PDF to generate corrections.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the MCQ corrector.
        
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
    
    def load_validation_results(self, file_path: str) -> Dict:
        """
        Load validation results from JSON file.
        
        Args:
            file_path: Path to validation results file
            
        Returns:
            Validation results dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading validation results: {e}")
            return {}
    
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

    def generate_correction_prompt(self, chunk_result: Dict) -> str:
        """
        Generate a prompt for Claude to correct questions based on validation results.
        
        Args:
            chunk_result: Chunk validation result dictionary
            
        Returns:
            Correction prompt
        """
        # Format all questions in the chunk with identified issues
        questions_text = []
        
        for question_result in chunk_result['questions']:
            original_question = question_result['original_question']
            issues = question_result['issues']
            
            q_text = f"Question {original_question['question_num']} ({original_question['page_ref']}): {original_question['question_text']}"
            
            for option in ['A', 'B', 'C', 'D']:
                if option in original_question['options']:
                    q_text += f"\n{option}. {original_question['options'][option]}"
            
            q_text += "\n\nExplanation:"
            
            for option in ['A', 'B', 'C', 'D']:
                if option in original_question['explanations']:
                    q_text += f"\n{option}. {original_question['explanations'][option]}"
            
            # Add identified issues
            if issues['has_issues']:
                q_text += "\n\nISSUES DETECTED:"
                
                if issues['multiple_correct_answers']:
                    correct_opts = original_question['correct_options']
                    q_text += f"\n- Multiple correct answers: {', '.join(correct_opts)}"
                
                if issues['no_correct_answers']:
                    q_text += "\n- No correct answers identified"
                
                if issues['cross_references']:
                    for ref in issues['cross_references']:
                        q_text += f"\n- Option {ref['option']} references correct option {ref['reference']}"
                
                if issues['forbidden_phrases']:
                    for phrase in issues['forbidden_phrases']:
                        q_text += f"\n- Option {phrase['option']} uses forbidden phrase: \"{phrase['phrase']}\""
                
                if issues['missing_abbreviation_expansion']:
                    abbrs = issues['missing_abbreviation_expansion']
                    q_text += f"\n- Missing abbreviation expansion for: {', '.join(abbrs)}"
                
                if issues['unclear_question']:
                    q_text += "\n- Question text may be unclear or too short"
            
            questions_text.append(q_text)
        
        combined_questions = "\n\n".join(questions_text)
        
        prompt = f"""You are a subject matter expert CORRECTING multiple-choice questions from a tax document. You must ACTIVELY REWRITE any problematic questions and explanations based on the detected issues.

QUESTIONS TO CORRECT WITH IDENTIFIED ISSUES:

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
    
    def parse_question_content(self, content: str, q_num: int, page_ref: str, chunk_idx: int) -> Optional[Dict]:
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

    def parse_correction_response(self, response_text: str, chunk_result: Dict) -> Dict:
        """
        Parse Claude's correction response to extract corrected questions.
        
        Args:
            response_text: Claude's response text
            chunk_result: Original chunk validation result
            
        Returns:
            Dictionary with corrected questions
        """
        # Create a copy of the chunk result for correction
        corrected_chunk = {
            'chunk_idx': chunk_result['chunk_idx'],
            'start_page': chunk_result['start_page'],
            'end_page': chunk_result['end_page'],
            'questions': []
        }
        
        # Remove markdown code blocks if present
        clean_response = re.sub(r'```', '', response_text).strip()
        
        # Find all corrected questions in the response
        question_pattern = r'Question\s+(\d+)\s+\(((?:Page|Pages)\s+[^)]+)\):\s*(.*?)(?=\n\nQuestion|\Z)'
        question_matches = re.findall(question_pattern, clean_response, re.DOTALL)
        
        for q_num_str, page_ref, q_content in question_matches:
            try:
                q_num = int(q_num_str)
                
                # Find the original question result
                original_question_result = next((q for q in chunk_result['questions'] 
                                               if q['question_num'] == q_num), None)
                
                if not original_question_result:
                    print(f"Warning: Could not find original Question {q_num} in chunk {chunk_result['chunk_idx']}.")
                    continue
                
                original_question = original_question_result['original_question']
                
                # Parse the corrected question content
                corrected_question = self.parse_question_content(
                    q_content.strip(), q_num, page_ref, chunk_result['chunk_idx']
                )
                
                if corrected_question:
                    # Track specific changes between original and corrected
                    corrected_question['changes'] = {
                        'question_text': {
                            'was_changed': original_question['question_text'] != corrected_question['question_text'],
                            'before': original_question['question_text'],
                            'after': corrected_question['question_text'],
                        },
                        'options_changed': {},
                        'explanations_changed': {}
                    }
                    
                    # Track option changes
                    all_option_keys = set(list(original_question['options'].keys()) + list(corrected_question['options'].keys()))
                    for opt in all_option_keys:
                        original_opt = original_question['options'].get(opt, "")
                        corrected_opt = corrected_question['options'].get(opt, "")
                        
                        corrected_question['changes']['options_changed'][opt] = {
                            'was_changed': original_opt != corrected_opt,
                            'before': original_opt,
                            'after': corrected_opt
                        }
                    
                    # Track explanation changes
                    all_expl_keys = set(list(original_question['explanations'].keys()) + list(corrected_question['explanations'].keys()))
                    for opt in all_expl_keys:
                        original_expl = original_question['explanations'].get(opt, "")
                        corrected_expl = corrected_question['explanations'].get(opt, "")
                        
                        corrected_question['changes']['explanations_changed'][opt] = {
                            'was_changed': original_expl != corrected_expl,
                            'before': original_expl,
                            'after': corrected_expl
                        }
                    
                    # Determine if anything was actually changed
                    any_changes = (
                        corrected_question['changes']['question_text']['was_changed'] or
                        any(change['was_changed'] for change in corrected_question['changes']['options_changed'].values()) or
                        any(change['was_changed'] for change in corrected_question['changes']['explanations_changed'].values())
                    )
                    
                    corrected_question['was_corrected'] = any_changes
                    corrected_question['page_numbers'] = original_question['page_numbers']
                    corrected_question['original_issues'] = original_question_result['issues']
                    corrected_chunk['questions'].append(corrected_question)
                else:
                    print(f"Warning: Could not parse corrected Question {q_num}.")
                    original_question['was_corrected'] = False
                    original_question['original_issues'] = original_question_result['issues']
                    corrected_chunk['questions'].append(original_question)
            
            except Exception as e:
                print(f"Error parsing corrected Question {q_num_str}: {e}")
        
        # If any questions are missing in the corrected set, use the originals
        for question_result in chunk_result['questions']:
            original_question = question_result['original_question']
            if not any(q['question_num'] == original_question['question_num'] for q in corrected_chunk['questions']):
                original_question['was_corrected'] = False
                original_question['original_issues'] = question_result['issues']
                corrected_chunk['questions'].append(original_question)
        
        # Sort questions by question number
        corrected_chunk['questions'].sort(key=lambda q: q['question_num'])
        
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

    def correct_chunk(self, chunk_result: Dict, pdf_path: str) -> Dict:
        """
        Correct all questions in a chunk using Claude with rate limiting.
        
        Args:
            chunk_result: Chunk validation result dictionary
            pdf_path: Path to the PDF file
            
        Returns:
            Corrected chunk dictionary
        """
        # Extract chunk pages as images
        pdf_pages = self.extract_pdf_chunk_as_images(pdf_path, chunk_result['start_page'], chunk_result['end_page'])
        
        if not pdf_pages:
            print(f"Warning: Could not extract pages for chunk {chunk_result['chunk_idx']}.")
            return {
                'chunk_idx': chunk_result['chunk_idx'],
                'start_page': chunk_result['start_page'],
                'end_page': chunk_result['end_page'],
                'questions': [q['original_question'] for q in chunk_result['questions']]
            }
        
        # Generate correction prompt
        prompt = self.generate_correction_prompt(chunk_result)
        
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
            
            print(f"Correcting Chunk {chunk_result['chunk_idx']} with {len(chunk_result['questions'])} questions...")
            
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
            
            # Parse correction response
            corrected_chunk = self.parse_correction_response(response.content[0].text, chunk_result)
            
            print(f"Completed correction of Chunk {chunk_result['chunk_idx']}")
            return corrected_chunk
            
        except Exception as e:
            print(f"Error correcting chunk {chunk_result['chunk_idx']}: {e}")
            
            # Sleep longer on error to recover from rate limits
            print(f"Sleeping for 60 seconds after error...")
            time.sleep(60)
            
            # Return original questions on error
            return {
                'chunk_idx': chunk_result['chunk_idx'],
                'start_page': chunk_result['start_page'],
                'end_page': chunk_result['end_page'],
                'questions': [q['original_question'] for q in chunk_result['questions']]
            }

    def generate_correction_report(self, corrected_data: Dict, output_dir: str):
        """
        Generate a detailed report of correction results.
        
        Args:
            corrected_data: Corrected questions data
            output_dir: Directory to save the report
        """
        # Count the number of questions corrected
        corrected_count = 0
        for chunk in corrected_data['chunks']:
            for question in chunk['questions']:
                if question.get('was_corrected', False):
                    corrected_count += 1
        
        report = [
            "# MCQ Correction Report",
            "",
            "## Summary",
            f"- Total questions: {corrected_data['total_questions']}",
            f"- Questions corrected: {corrected_count}",
            f"- Percentage corrected: {(corrected_count / corrected_data['total_questions'] * 100):.1f}%",
            "",
            "## Token Usage",
            f"- Input tokens: {self.total_input_tokens:,}",
            f"- Output tokens: {self.total_output_tokens:,}",
            f"- Total tokens: {(self.total_input_tokens + self.total_output_tokens):,}",
            "",
            "## Correction by Chunk"
        ]
        
        # Add details for each chunk
        for chunk in corrected_data['chunks']:
            chunk_corrected = sum(1 for q in chunk['questions'] if q.get('was_corrected', False))
            
            report.append(f"### Chunk {chunk['chunk_idx']} (Pages {chunk['start_page']} to {chunk['end_page']})")
            report.append(f"- Questions in chunk: {len(chunk['questions'])}")
            report.append(f"- Questions corrected: {chunk_corrected}")
            report.append("")
        
        # Write report to file
        report_path = os.path.join(output_dir, "correction_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
        print(f"Correction report saved to {report_path}")

    def save_detailed_changes_report(self, corrected_data: Dict, output_dir: str):
        """
        Generate a detailed report of specific changes made to each question.
        
        Args:
            corrected_data: Corrected questions data
            output_dir: Directory to save the report
        """
        changes_report = []
        changes_found = False
        
        for chunk in corrected_data['chunks']:
            for question in chunk['questions']:
                if question.get('was_corrected', False):
                    changes_found = True
                    q_changes = f"## Question {question['question_num']} (Chunk {chunk['chunk_idx']})\n\n"
                    
                    # Original issues
                    if 'original_issues' in question:
                        issues = question['original_issues']
                        issue_list = []
                        
                        if issues['multiple_correct_answers']:
                            issue_list.append("- Multiple correct answers")
                        
                        if issues['no_correct_answers']:
                            issue_list.append("- No correct answers identified")
                        
                        if issues['cross_references']:
                            issue_list.append("- Explanations referenced correct answer")
                        
                        if issues['forbidden_phrases']:
                            issue_list.append("- Used forbidden phrases")
                        
                        if issues['missing_abbreviation_expansion']:
                            issue_list.append("- Missing abbreviation expansion")
                        
                        if issues['unclear_question']:
                            issue_list.append("- Unclear question text")
                        
                        if issue_list:
                            q_changes += "### Original Issues\n\n"
                            q_changes += "\n".join(issue_list) + "\n\n"
                    
                    # Question text changes
                    if question['changes']['question_text']['was_changed']:
                        q_changes += "### Question Text Changed\n\n"
                        q_changes += f"**Before:** {question['changes']['question_text']['before']}\n\n"
                        q_changes += f"**After:** {question['changes']['question_text']['after']}\n\n"
                    
                    # Option changes
                    option_changes = [opt for opt, change in question['changes']['options_changed'].items() 
                                    if change['was_changed']]
                    if option_changes:
                        q_changes += "### Options Changed\n\n"
                        for opt in sorted(option_changes):
                            change = question['changes']['options_changed'][opt]
                            q_changes += f"**Option {opt}:**\n"
                            q_changes += f"- Before: {change['before']}\n"
                            q_changes += f"- After: {change['after']}\n\n"
                    
                    # Explanation changes
                    expl_changes = [opt for opt, change in question['changes']['explanations_changed'].items() 
                                   if change['was_changed']]
                    if expl_changes:
                        q_changes += "### Explanations Changed\n\n"
                        for opt in sorted(expl_changes):
                            change = question['changes']['explanations_changed'][opt]
                            q_changes += f"**Explanation {opt}:**\n"
                            q_changes += f"- Before: {change['before']}\n"
                            q_changes += f"- After: {change['after']}\n\n"
                    
                    changes_report.append(q_changes)
        
        # If changes were found, save to a changes report file
        if changes_found:
            changes_path = os.path.join(output_dir, "detailed_changes.md")
            with open(changes_path, 'w', encoding='utf-8') as f:
                f.write("# Detailed Changes Report\n\n")
                f.write("\n".join(changes_report))
            print(f"Detailed changes report saved to {changes_path}")

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
            'chunks': [],
            'total_questions': corrected_data['total_questions']
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
                    'was_corrected': question.get('was_corrected', False)
                }
                
                # Include change information if the question was corrected
                if question.get('was_corrected', False):
                    json_question['changes'] = question.get('changes', {})
                
                json_chunk['questions'].append(json_question)
            
            json_data['chunks'].append(json_chunk)
        
        json_path = os.path.join(output_dir, "corrected_questions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        print(f"Structured data saved to {json_path}")

    def correct_mcqs(self, validation_file: str, pdf_path: str, output_dir: str = "correction_output"):
        """
        Main method to correct MCQs using Claude based on validation results.
        
        Args:
            validation_file: Path to the validation results file
            pdf_path: Path to the source PDF
            output_dir: Directory to save output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading validation results from {validation_file}")
        validation_results = self.load_validation_results(validation_file)
        
        if not validation_results or 'chunks' not in validation_results:
            print("Error: No validation results found or invalid format.")
            return
            
        print(f"Found {validation_results['total_questions']} questions in {len(validation_results['chunks'])} chunks")
        
        # Correct each chunk
        corrected_data = {
            'chunks': [],
            'total_questions': validation_results['total_questions']
        }
        
        for chunk_result in validation_results['chunks']:
            corrected_chunk = self.correct_chunk(chunk_result, pdf_path)
            corrected_data['chunks'].append(corrected_chunk)
        
        # Generate correction report
        print("Generating correction report...")
        self.generate_correction_report(corrected_data, output_dir)
        
        # Save detailed changes report
        print("Generating detailed changes report...")
        self.save_detailed_changes_report(corrected_data, output_dir)
        
        # Save corrected questions
        print("Saving corrected questions...")
        self.save_corrected_questions(corrected_data, output_dir)
        
        print(f"Correction complete. Results saved to {output_dir}/")
        print(f"Token usage: {self.total_input_tokens:,} input + {self.total_output_tokens:,} output = {(self.total_input_tokens + self.total_output_tokens):,} tokens")

def main():
    parser = argparse.ArgumentParser(description='Correct MCQ questions based on validation results')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation results file')
    parser.add_argument('--pdf', type=str, required=True, help='Path to source PDF file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='correction_output', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', 
                       help='Claude model to use (must be vision-capable)')
    
    args = parser.parse_args()
    
    # Initialize corrector and run correction
    corrector = MCQCorrector(api_key=args.api_key, model=args.model)
    corrector.correct_mcqs(
        validation_file=args.validation,
        pdf_path=args.pdf,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()