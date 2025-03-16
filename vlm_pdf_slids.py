import os
import re
import math
import base64
import argparse
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
		A: why it's right or wrong,
		B: why it's right or wrong,
		C: why it's right or wrong,
		D: why it's right or wrong,	
"""

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
        
        prompt = f"""You are a subject matter expert who deeply understands this educational topic. You need to create thoughtful quiz questions that test key concepts from this material.

I'm providing pages {first_page} to {last_page} from a course slide deck as images.

IMPORTANT INSTRUCTIONS:
1. IGNORE any title slides, agenda slides, or introduction content that doesn't contain actual educational material
2. Pay special attention to tables, graphs, diagrams, and other visual elements
3. Generate {num_questions} challenging but fair multiple-choice questions that test understanding of important concepts
4. Each question must:
   - Be directly based on the educational content (include page numbers beside each question)
   - Have 4 answer options with exactly one correct answer
   - Include an explanation that demonstrates deep subject knowledge
   - Include visual elements like diagrams, tables, and charts in your analysis
5. Explanation should be like this:
    {explanation}

critical: your questions and explanations must read as if written by a subject matter expert who genuinely understands the topic.Questions and explainations should be like they are final examination questions, do not use phrases like:
- "According to the slides..."
- "In this presentation..."
- "The slide mentions..."
- "As stated in page X..."
- "Based on the material provided..." etc

Instead, explain concepts authoritatively as established knowledge in the field. The questions should feel like they were hand-crafted by someone with expert understanding of the subject.

After the questions, provide a brief 2-3 sentence summary of the key concepts covered in this section.
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
        
        prompt = f"""You are a subject matter expert who deeply understands this educational topic. You need to create thoughtful quiz questions that test key concepts from this material.

I'm providing pages {first_page} to {last_page} from a course slide deck as images.

Context from previous sections:
{previous_summaries}

IMPORTANT INSTRUCTIONS:
1. First, thoroughly read and understand how this section builds on previous knowledge
2. Pay special attention to tables, graphs, diagrams, and other visual elements
3. Generate {num_questions} challenging but fair multiple-choice questions that test understanding of NEW concepts in this section
4. Each question must:
   - Be directly based on the educational content from THIS section (include page numbers beside each question)
   - Have 4 answer options with exactly one correct answer
   - Include an explanation that demonstrates deep subject knowledge
   - Include visual elements like diagrams, tables, and charts in your analysis
   - Feel connected to the broader learning journey
5. Explanation should be like this:
    {explanation}

CRITICAL: Your questions and explanations must read as if written by a subject matter expert who genuinely understands the entire topic.Questions and explainations should be like they are final examination questions, Do NOT use phrases like:
- "According to the slides..."
- "In this presentation..."
- "The slide mentions..."
- "As stated in page X..."
- "Based on the material provided..." etc

Instead, explain concepts authoritatively as established knowledge in the field. The questions should feel like they were hand-crafted by someone with expert understanding of the subject who is carefully building upon previously established concepts.

After the questions, provide a brief 2-3 sentence summary of the key concepts covered in this section.
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
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            print(f"Response received - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            print(f"Running total - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}, Total: {self.total_input_tokens + self.total_output_tokens}")
            
            return {
                "response": response.content[0].text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
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
                "output_tokens": 0
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
        Process the entire PDF, chunk it, and generate questions using vision models.
        
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
        combined_summaries = ""
        token_usage = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)} (pages {chunk[0]['page_num']} to {chunk[-1]['page_num']})...")
            print(f"Chunk contains {len(chunk)} pages")
            
            is_first_chunk = (i == 0)
            
            # Generate questions for this chunk
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
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0)
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
            
            # Write individual chunk result to file
            with open(f"{output_dir}/chunk_{i+1}_questions.txt", "w", encoding="utf-8") as f:
                f.write(f"Questions for pages {chunk[0]['page_num']} to {chunk[-1]['page_num']}:\n\n")
                f.write(result["response"])
                
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
                f.write(f"    Total tokens:  {chunk_data['total_tokens']:,}\n\n")
            
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
    parser = argparse.ArgumentParser(description='Generate quiz questions from PDF slides using Vision Models')
    parser.add_argument('--file', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', 
                        help='Claude model to use (must be vision-capable)')
    parser.add_argument('--target-pages', type=int, default=20, 
                       help='Target number of pages per chunk (default: 30)')
    parser.add_argument('--skip-pages', type=int, default=3,
                       help='Number of initial pages to skip (default: 3)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution for page rendering (default: 300)')
    parser.add_argument('--optimize-tokens', action='store_true', default=True,
                       help='Optimize images to reduce token usage')
    parser.add_argument('--high-quality', action='store_true',
                       help='Use higher quality images (more tokens)')
    
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