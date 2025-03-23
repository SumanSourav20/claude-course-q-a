import os
import re
import math
import docx
import anthropic
import argparse
from typing import List, Dict, Tuple, Optional

class TranscriptProcessor:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the transcript processor with API key and model.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.summaries = []
        self.questions = []

    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a docx file including timestamps.
        
        Args:
            file_path: Path to the docx file
            
        Returns:
            String containing the full transcript text
        """
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def chunk_transcript(self, transcript: str, num_chunks: int = 30) -> List[str]:
        """
        Divide the transcript into approximately equal chunks.
        
        Args:
            transcript: Full transcript text
            num_chunks: Target number of chunks
            
        Returns:
            List of chunked transcript sections
        """
        lines = transcript.split('\n')
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Calculate lines per chunk (approximate)
        lines_per_chunk = math.ceil(len(lines) / num_chunks)
        
        chunks = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = '\n'.join(lines[i:i + lines_per_chunk])
            chunks.append(chunk)
            
        return chunks
    
    def extract_timestamp_range(self, chunk: str) -> Tuple[str, str]:
        """
        Extract the first and last timestamp from a chunk.
        
        Args:
            chunk: Transcript chunk
            
        Returns:
            Tuple of (first_timestamp, last_timestamp)
        """
        # Common timestamp patterns (adjust as needed for your transcript format)
        timestamp_pattern = r'(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})'
        
        timestamps = re.findall(timestamp_pattern, chunk)
        
        if timestamps:
            return timestamps[0], timestamps[-1]
        return "00:00", "00:00"  # Default if no timestamps found

    def generate_first_chunk_prompt(self, chunk: str) -> str:
        """
        Generate prompt for the first chunk, instructing to avoid irrelevant content.
        
        Args:
            chunk: First transcript chunk
            
        Returns:
            Formatted prompt for Claude
        """
        first_timestamp, last_timestamp = self.extract_timestamp_range(chunk)
        
        prompt = f"""You are an expert in creating educational quiz questions. I have a course video transcript that I need to generate MCQ questions from.

This is the first part of the transcript (timestamps {first_timestamp} to {last_timestamp}). 

IMPORTANT: Ignore any content about course logistics, how to pass the course, certificate information, or instructor introductions. Only focus on the actual course content and subject matter.

Please:
1. Understand the main topic being taught
2. Generate 10 multiple-choice questions (MCQs) based only on the educational content
3. For each question, provide 4 options with only one correct answer
4. Include detailed explanations for why the correct answer is right and why the others are wrong
5. Make sure questions and answers directly relate to content in the transcript
6. Include the approximate timestamp range where the question content appears
7. Create a brief summary (2-3 sentences) of the key concepts covered in this section

Do not use phrases like "according to the transcript" or "the instructor mentions" in your explanations. Write questions and explanations as if you naturally understood the material.

Here is the transcript:
{chunk}
"""
        return prompt

    def generate_subsequent_chunk_prompt(self, chunk: str, previous_summaries: str) -> str:
        """
        Generate prompt for subsequent chunks, including previous summaries for context.
        
        Args:
            chunk: Current transcript chunk
            previous_summaries: Combined summaries from previous chunks
            
        Returns:
            Formatted prompt for Claude
        """
        first_timestamp, last_timestamp = self.extract_timestamp_range(chunk)
        
        num_questions = 5  # Default for subsequent chunks
        
        prompt = f"""You are an expert in creating educational quiz questions. I have a course video transcript that I need to generate MCQ questions from.

This is a continuation of a course transcript (timestamps {first_timestamp} to {last_timestamp}).

Context from previous sections:
{previous_summaries}

Please:
1. Understand the main topic being taught in this section
2. Generate {num_questions} multiple-choice questions (MCQs) based only on the educational content in this section
3. For each question, provide 4 options with only one correct answer
4. Include detailed explanations for why the correct answer is right and why the others are wrong
5. Make sure questions and answers directly relate to content in this section of the transcript
6. Include the approximate timestamp range where the question content appears
7. Create a brief summary (2-3 sentences) of the key concepts covered in this section

Do not use phrases like "according to the transcript" or "the instructor mentions" in your explanations. Write questions and explanations as if you naturally understood the material.

Here is the transcript section:
{chunk}
"""
        return prompt

    def generate_questions_from_chunk(self, chunk: str, is_first_chunk: bool, previous_summaries: Optional[str] = None) -> Dict:
        """
        Generate MCQ questions from a transcript chunk using Claude.
        
        Args:
            chunk: Transcript chunk
            is_first_chunk: Whether this is the first chunk
            previous_summaries: Combined summaries from previous chunks
            
        Returns:
            Dict containing generated questions and summary
        """
        if is_first_chunk:
            prompt = self.generate_first_chunk_prompt(chunk)
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
            r"Brief summary:(.*?)(?=\n\n|\Z)"
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no clear summary section, take the last paragraph
        paragraphs = response.split("\n\n")
        for para in reversed(paragraphs):
            if len(para.split()) > 10 and "question" not in para.lower():
                return para.strip()
                
        return "No clear summary found."

    def process_transcript(self, file_path: str, output_dir: str = "quiz_output"):
        """
        Process the entire transcript, chunk it, and generate questions.
        
        Args:
            file_path: Path to the transcript docx file
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract and chunk the transcript
        transcript = self.extract_text_from_docx(file_path)
        chunks = self.chunk_transcript(transcript)
        
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
                combined_summaries if not is_first_chunk else None
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
    parser = argparse.ArgumentParser(description='Generate quiz questions from course transcript')
    parser.add_argument('--file', type=str, required=True, help='Path to transcript docx file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output', help='Output directory')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', help='Claude model to use')
    parser.add_argument('--chunks', type=int, default=30, help='Number of chunks to divide transcript into')
    
    args = parser.parse_args()
    
    processor = TranscriptProcessor(api_key=args.api_key, model=args.model)
    processor.process_transcript(args.file, args.output_dir)