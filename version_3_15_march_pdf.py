import os
import re
import math
import fitz  # PyMuPDF
import anthropic
import argparse
from typing import List, Dict, Tuple, Optional, Any

explanation = """
	explanation: 
		A: why it's right or wrong,
		B: why it's right or wrong,
		C: why it's right or wrong,
		D: why it's right or wrong,	
"""

class PDFProcessor:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the PDF processor wh API key and model.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.summaries = []
        self.questions = []

    def extract_text_from_pdf(self, file_path: str, skip_pages: int = 3) -> Tuple[str, int]:
        """
        Extract text from a PDF file and get the actual page count.
        
        Args:
            file_path: Path to the PDF file
            skip_pages: Number of initial pages to skip (default: 3)
            
        Returns:
            Tuple of (full_text_content, page_count_excluding_skipped)
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
            
            # Extract text from each page, skipping the first skip_pages
            content = []
            for page_num in range(effective_start_page, total_pages):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                if text.strip():  # Only include non-empty pages
                    # Add page number reference to help with tracking
                    content.append(f"[Page {page_num+1}]\n{text}")
            
            pdf_document.close()
            return "\n\n".join(content), effective_pages
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return "", 0

    def chunk_pdf_content(self, content: str, total_pages: int, target_pages_per_chunk: int = 30) -> List[str]:
        """
        Divide the PDF content into chunks based on page markers.
        
        Args:
            content: Full PDF text content with page markers
            total_pages: Total number of effective pages
            target_pages_per_chunk: Target number of pages per chunk
            
        Returns:
            List of chunked PDF content sections
        """
        # If total pages is close to target, keep as single chunk
        if total_pages <= target_pages_per_chunk + 5:
            return [content]
        
        # Calculate optimal number of chunks
        num_chunks = max(1, round(total_pages / target_pages_per_chunk))
        
        # If dividing would make chunks too small, reduce number of chunks
        pages_per_chunk = total_pages / num_chunks
        if pages_per_chunk < 15 and num_chunks > 1:
            num_chunks -= 1
            pages_per_chunk = total_pages / num_chunks
        
        # Find page markers to split by
        page_markers = re.findall(r'\[Page \d+\]', content)
        
        # If no page markers found, use a simple text split
        if not page_markers:
            # Fallback to splitting by paragraph
            paragraphs = content.split('\n\n')
            paras_per_chunk = math.ceil(len(paragraphs) / num_chunks)
            
            chunks = []
            for i in range(0, len(paragraphs), paras_per_chunk):
                chunk = '\n\n'.join(paragraphs[i:i + paras_per_chunk])
                chunks.append(chunk)
                
            print(f"Page markers not found. Split into {len(chunks)} chunks based on paragraphs.")
            return chunks
        
        # Calculate which page markers to use as chunk boundaries
        pages_per_chunk = math.ceil(total_pages / num_chunks)
        
        # Get the page numbers from markers
        page_numbers = [int(re.search(r'\[Page (\d+)\]', marker).group(1)) for marker in page_markers]
        
        # Find chunk boundary pages
        chunk_boundaries = []
        for i in range(1, num_chunks):
            target_page = i * pages_per_chunk
            # Find the closest page marker
            closest_idx = min(range(len(page_numbers)), key=lambda j: abs(page_numbers[j] - target_page))
            chunk_boundaries.append(page_markers[closest_idx])
        
        # Split the content by these boundaries
        chunks = []
        remaining_content = content
        for boundary in chunk_boundaries:
            parts = remaining_content.split(boundary, 1)
            if len(parts) == 2:
                chunks.append(parts[0])
                remaining_content = boundary + parts[1]
            else:
                print(f"Warning: Could not split at boundary {boundary}")
        
        # Add the last chunk
        chunks.append(remaining_content)
        
        print(f"Split into {len(chunks)} chunks based on page boundaries")
        
        return chunks
    
    def extract_page_range(self, chunk: str) -> Tuple[int, int]:
        """
        Extract the first and last page number from a chunk.
        
        Args:
            chunk: PDF content chunk
            
        Returns:
            Tuple of (first_page, last_page)
        """
        page_numbers = re.findall(r'\[Page (\d+)\]', chunk)
        
        if page_numbers:
            return int(page_numbers[0]), int(page_numbers[-1])
        return 0, 0  # Default if no page numbers found

    def generate_first_chunk_prompt(self, chunk: str, total_chunks: int) -> str:
        """
        Generate prompt for the first chunk, instructing to avoid irrelevant content.
        
        Args:
            chunk: First PDF content chunk
            total_chunks: Total number of chunks
            
        Returns:
            Formatted prompt for Claude
        """
        first_page, last_page = self.extract_page_range(chunk)
        
        # If only one chunk exists, generate 10 questions, otherwise 5
        num_questions = 10 if total_chunks == 1 else 5
        
        prompt = f"""You are a subject matter expert who deeply understands this educational topic. You need to create thoughtful quiz questions that test key concepts from this material.

This section covers content from pages {first_page} to {last_page} of a course slide deck.

IMPORTANT INSTRUCTIONS:
1. IGNORE any title slides, agenda slides, or introduction content that doesn't contain actual educational material
2. First, thoroughly read and understand the core educational concepts being taught
3. Generate {num_questions} challenging but fair multiple-choice questions that test understanding of important concepts
4. Each question must:
   - Be directly based on the educational content (include page numbers beside each question)
   - Have 4 answer options with exactly one correct answer
   - Include an explanation that demonstrates deep subject knowledge
5. Explanation should be like this:
    {explanation}

CRITICAL: Your questions and explanations must read as if written by a subject matter expert who genuinely understands the topic. Do NOT use phrases like:
- "According to the slides..."
- "In this presentation..."
- "The slide mentions..."
- "As stated in page X..."
- "Based on the material provided..."

Instead, explain concepts authoritatively as established knowledge in the field. The questions should feel like they were hand-crafted by someone with expert understanding of the subject.

After the questions, provide a brief 2-3 sentence summary of the key concepts covered in this section.

Here is the content:
{chunk}
"""
        return prompt

    def generate_subsequent_chunk_prompt(self, chunk: str, previous_summaries: str) -> str:
        """
        Generate prompt for subsequent chunks, including previous summaries for context.
        
        Args:
            chunk: Current PDF content chunk
            previous_summaries: Combined summaries from previous chunks
            
        Returns:
            Formatted prompt for Claude
        """
        first_page, last_page = self.extract_page_range(chunk)
        
        num_questions = 5  # Default for subsequent chunks
        
        prompt = f"""You are a subject matter expert who deeply understands this educational topic. You need to create thoughtful quiz questions that test key concepts from this material.

This section covers content from pages {first_page} to {last_page} of a course slide deck.

Context from previous sections:
{previous_summaries}

IMPORTANT INSTRUCTIONS:
1. First, thoroughly read and understand how this section builds on previous knowledge
2. Generate {num_questions} challenging but fair multiple-choice questions that test understanding of NEW concepts in this section
3. Each question must:
   - Be directly based on the educational content from THIS section (include page numbers beside each question)
   - Have 4 answer options with exactly one correct answer
   - Include an explanation that demonstrates deep subject knowledge
   - Feel connected to the broader learning journey
4. Explanation should be like this:
    {explanation}

CRITICAL: Your questions and explanations must read as if written by a subject matter expert who genuinely understands the entire topic. Do NOT use phrases like:
- "According to the slides..."
- "In this presentation..."
- "The slide mentions..."
- "As stated in page X..."
- "Based on the material provided..."

Instead, explain concepts authoritatively as established knowledge in the field. The questions should feel like they were hand-crafted by someone with expert understanding of the subject who is carefully building upon previously established concepts.

After the questions, provide a brief 2-3 sentence summary of the key concepts covered in this section.

Here is the content for this section:
{chunk}
"""
        return prompt

    def generate_questions_from_chunk(self, chunk: str, is_first_chunk: bool, 
                                      previous_summaries: Optional[str] = None, total_chunks: int = 1) -> Dict:
        """
        Generate MCQ questions from a PDF chunk using Claude.
        
        Args:
            chunk: PDF content chunk
            is_first_chunk: Whether this is the first chunk
            previous_summaries: Combined summaries from previous chunks
            total_chunks: Total number of chunks in the document
            
        Returns:
            Dict containing generated questions and summary
        """
        if is_first_chunk:
            prompt = self.generate_first_chunk_prompt(chunk, total_chunks)
        else:
            prompt = self.generate_subsequent_chunk_prompt(chunk, previous_summaries)
            
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {"response": response.content[0].text}

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

    def process_pdf(self, file_path: str, output_dir: str = "quiz_output", target_pages_per_chunk: int = 30, skip_pages: int = 3):
        """
        Process the entire PDF, chunk it, and generate questions.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save output files
            target_pages_per_chunk: Target number of pages per chunk
            skip_pages: Number of initial pages to skip
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract PDF content and get page count
        content, effective_pages = self.extract_text_from_pdf(file_path, skip_pages)
        if not content:
            print("Error: Could not extract content from PDF.")
            return
            
        print(f"Successfully extracted content from {effective_pages} pages of PDF.")
        
        # Chunk the content
        chunks = self.chunk_pdf_content(content, effective_pages, target_pages_per_chunk)
        
        all_questions = []
        combined_summaries = ""
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            is_first_chunk = (i == 0)
            
            # Generate questions for this chunk
            result = self.generate_questions_from_chunk(
                chunk, 
                is_first_chunk, 
                combined_summaries if not is_first_chunk else None,
                total_chunks=len(chunks)
            )
            
            # Extract and save summary for context in next chunks
            summary = self.extract_summary_from_response(result["response"])
            chunk_summary = f"Chunk {i+1}: {summary}"
            self.summaries.append(chunk_summary)
            
            # Update combined summaries for next chunk
            if combined_summaries:
                combined_summaries += f"\n\n{chunk_summary}"
            else:
                combined_summaries = chunk_summary
                
            # Save the questions for this chunk
            all_questions.append(result["response"])
            
            # Write individual chunk result to file
            with open(f"{output_dir}/chunk_{i+1}_questions.txt", "w", encoding="utf-8") as f:
                f.write(result["response"])
                
        # Write all questions to a single file
        with open(f"{output_dir}/all_questions.txt", "w", encoding="utf-8") as f:
            for i, questions in enumerate(all_questions):
                f.write(f"=== CHUNK {i+1} ===\n\n")
                f.write(questions)
                f.write("\n\n")
                
        # Write summaries to a file
        with open(f"{output_dir}/summaries.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.summaries))
            
        print(f"Processing complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate quiz questions from PDF slides')
    parser.add_argument('--file', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output_pdf', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', help='Claude model to use')
    parser.add_argument('--target-pages', type=int, default=20, 
                       help='Target number of pages per chunk (default: 30)')
    parser.add_argument('--skip-pages', type=int, default=3,
                       help='Number of initial pages to skip (default: 3)')
    
    args = parser.parse_args()
    
    processor = PDFProcessor(api_key=args.api_key, model=args.model)
    processor.process_pdf(args.file, args.output_dir, args.target_pages, args.skip_pages)