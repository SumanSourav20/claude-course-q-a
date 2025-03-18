import os
import re
import anthropic
import argparse
from typing import List, Dict, Tuple, Optional

class TimestampCorrector:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the timestamp corrector with API key and model.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_original_chunks(self, output_dir: str) -> List[Tuple[int, str]]:
        """
        Find all question files in the output directory to determine how many chunks were used.
        
        Args:
            output_dir: Directory containing generated question files
            
        Returns:
            List of tuples with (chunk_number, file_path)
        """
        chunk_files = []
        
        # Find all chunk_*_questions.txt files
        for file in os.listdir(output_dir):
            match = re.match(r'chunk_(\d+)_questions\.txt', file)
            if match:
                chunk_num = int(match.group(1))
                file_path = os.path.join(output_dir, file)
                chunk_files.append((chunk_num, file_path))
                
        # Sort by chunk number
        chunk_files.sort(key=lambda x: x[0])
        
        return chunk_files
        
    def read_chunk_file(self, transcript_file: str, chunk_num: int, target_pages_per_chunk: int = 30) -> str:
        """
        Read the original content from the transcript for the specified chunk.
        
        Args:
            transcript_file: Path to the transcript docx file
            chunk_num: The chunk number to extract
            target_pages_per_chunk: Target number of pages per chunk (to match original chunking)
            
        Returns:
            Content of the specified chunk
        """
        # Use the same chunking logic as TranscriptProcessor
        # Import here to avoid circular imports
        import sys
        import os
        
        # Extract directory from the script file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Add the script directory to the system path if it's not already there
        if script_dir not in sys.path:
            sys.path.append(script_dir)
            
        # Now import the TranscriptProcessor
        from version_3_14_march import TranscriptProcessor
        
        # Create dummy processor (API key not needed for chunking)
        processor = TranscriptProcessor(api_key="dummy")
        
        # Extract and chunk the transcript using the same method as the original
        transcript, total_pages = processor.extract_text_from_docx(transcript_file)
        chunks = processor.chunk_transcript(transcript, total_pages, target_pages_per_chunk)
        
        # Check if chunk_num is valid
        if chunk_num <= 0 or chunk_num > len(chunks):
            raise ValueError(f"Invalid chunk number: {chunk_num}. Total chunks: {len(chunks)}")
            
        return chunks[chunk_num - 1]  # Chunk numbers are 1-based

    def read_question_file(self, file_path: str) -> str:
        """
        Read the generated questions file.
        
        Args:
            file_path: Path to the question file
            
        Returns:
            Content of the question file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_questions_with_timestamps(self, content: str) -> List[Dict]:
        """
        Extract questions and their timestamps from the generated content.
        
        Args:
            content: Content of the question file
            
        Returns:
            List of dictionaries containing questions and their timestamps
        """
        # First try the pattern with explicit timestamps in parentheses
        question_pattern = r'(Question\s+\d+)(?:\s+\(([^)]+)\))?([\s\S]+?)(?=Question\s+\d+|$)'
        matches = re.findall(question_pattern, content, re.IGNORECASE)
        
        # If no matches with parenthesized timestamps, try more flexible pattern
        if not matches or all(not match[1] for match in matches):
            # Look for timestamps in other formats - e.g., timestamp might be on a line after question number
            # This pattern looks for question numbers followed by timestamp-like patterns
            alt_pattern = r'(Question\s+\d+)[\s\S]*?(\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?)([\s\S]+?)(?=Question\s+\d+|$)'
            matches = re.findall(alt_pattern, content, re.IGNORECASE)
        
        questions = []
        for match in matches:
            question_num = match[0].strip()
            timestamp = match[1].strip() if match[1] else "No timestamp found"
            question_content = match[2].strip()
            
            # If still no timestamp found, look for it within the question content
            if timestamp == "No timestamp found":
                # Look for timestamp pattern within the question content
                time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?)'
                time_matches = re.search(time_pattern, question_content)
                
                if time_matches:
                    timestamp = time_matches.group(1)
            
            questions.append({
                "question_num": question_num,
                "timestamp": timestamp,
                "content": question_content
            })
        
        print(f"Extracted {len(questions)} questions with timestamps")
        return questions

    def validate_timestamp(self, chunk: str, timestamp: str) -> bool:
        """
        Check if the timestamp is actually present in the chunk.
        
        Args:
            chunk: Transcript chunk
            timestamp: Timestamp to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Extract time ranges from timestamp (e.g., "10:15 - 12:30")
        time_ranges = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)', timestamp)
        
        if not time_ranges:
            return False
        
        # Check if at least one time in the range appears in the chunk
        for time in time_ranges:
            if time in chunk:
                return True
                
        return False

    def correct_timestamps(self, chunk: str, questions: List[Dict]) -> List[Dict]:
        """
        Generate corrected timestamps for questions using Claude.
        
        Args:
            chunk: Transcript chunk
            questions: List of questions with potentially incorrect timestamps
            
        Returns:
            List of questions with corrected timestamps
        """
        corrected_questions = []
        
        for question in questions:
            # Skip if timestamp is already valid
            if self.validate_timestamp(chunk, question["timestamp"]):
                corrected_questions.append(question)
                continue
                
            # Prepare prompt for Claude to find correct timestamp
            prompt = f"""You are tasked with finding the precise timestamp in a transcript where a specific concept is discussed. 

The following is a question from an educational quiz:

{question["question_num"]}
{question["content"]}

The current timestamp provided ({question["timestamp"]}) may be incorrect. 

Please carefully examine the transcript below and identify the EXACT timestamps (in format HH:MM or HH:MM:SS) where the information needed to answer this question is discussed. 

Return ONLY the correct timestamp range in this format: "HH:MM - HH:MM" or a single timestamp if it's a brief mention.

DO NOT include any explanations or other text in your response - just return the timestamp range.

Transcript:
{chunk}
"""
            
            # Get corrected timestamp from Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract timestamp from Claude's response
            corrected_timestamp = response.content[0].text.strip()
            
            # Check if Claude returned a valid timestamp format
            timestamp_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)'
            single_timestamp_pattern = r'^(\d{1,2}:\d{2}(?::\d{2})?)$'
            
            timestamp_match = re.search(timestamp_pattern, corrected_timestamp)
            single_match = re.search(single_timestamp_pattern, corrected_timestamp)
            
            if timestamp_match or single_match:
                # Timestamp format valid
                question["original_timestamp"] = question["timestamp"]
                question["timestamp"] = corrected_timestamp
            else:
                # If Claude didn't return a valid timestamp, keep original but mark it
                question["timestamp"] = f"{question['timestamp']} (unable to validate)"
                
            corrected_questions.append(question)
            
        return corrected_questions

    def update_question_file(self, file_path: str, original_content: str, corrected_questions: List[Dict]) -> None:
        """
        Update the question file with corrected timestamps.
        
        Args:
            file_path: Path to the question file
            original_content: Original content of the file
            corrected_questions: List of questions with corrected timestamps
        """
        updated_content = original_content
        
        for question in corrected_questions:
            if "original_timestamp" in question:
                # Try different patterns for replacement
                
                # Pattern 1: Question number followed directly by timestamp in parentheses
                pattern1 = f'{question["question_num"]} \\({re.escape(question["original_timestamp"])}\\)'
                replacement1 = f'{question["question_num"]} ({question["timestamp"]})'
                
                # Pattern 2: Timestamp appears on its own line
                pattern2 = f'{question["original_timestamp"]}'
                replacement2 = f'{question["timestamp"]}'
                
                # Try pattern 1 first
                new_content = re.sub(pattern1, replacement1, updated_content)
                
                # If content wasn't changed, try pattern 2
                if new_content == updated_content:
                    updated_content = re.sub(pattern2, replacement2, updated_content)
                else:
                    updated_content = new_content
        
        # Write the updated content to a corrected version
        corrected_path = file_path.replace('.txt', '_corrected.txt')
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        print(f"Corrected timestamps saved to {corrected_path}")
        
        # Don't modify the original file to preserve the original output
        # If you want to modify the original too, uncomment these lines:
        # with open(file_path, "w", encoding="utf-8") as f:
        #     f.write(updated_content)

    def process_all_chunks(self, transcript_file: str, output_dir: str, target_pages_per_chunk: int = 30):
        """
        Process all chunks and their corresponding question files.
        
        Args:
            transcript_file: Path to the original transcript docx file
            output_dir: Directory containing generated question files
            target_pages_per_chunk: Target number of pages per chunk (to match original chunking)
        """
        # Find all question files to determine how many chunks were created
        chunk_files = self.get_original_chunks(output_dir)
        
        print(f"Found {len(chunk_files)} chunk files in {output_dir}")
        
        # Create validation report file
        report_path = os.path.join(output_dir, "timestamp_validation_report.txt")
        with open(report_path, "w", encoding="utf-8") as report:
            report.write("# Timestamp Validation and Correction Report\n\n")
            
            # Process each chunk file
            for chunk_num, question_file in chunk_files:
                print(f"Processing chunk {chunk_num}...")
                
                # Get the original chunk content using the same chunking logic
                try:
                    chunk = self.read_chunk_file(transcript_file, chunk_num, target_pages_per_chunk)
                except Exception as e:
                    report.write(f"## Chunk {chunk_num}: Error reading original chunk: {str(e)}\n\n")
                    print(f"Error reading chunk {chunk_num}: {str(e)}")
                    continue
                
                report.write(f"## Chunk {chunk_num}:\n\n")
                
                # Read questions file
                content = self.read_question_file(question_file)
                
                # Extract questions with timestamps
                questions = self.extract_questions_with_timestamps(content)
                
                if not questions:
                    report.write("No questions with timestamps found in this chunk.\n\n")
                    continue
                
                # Validate and correct timestamps
                corrected_questions = self.correct_timestamps(chunk, questions)
                
                # Update report with validation results
                for question in corrected_questions:
                    if "original_timestamp" in question:
                        report.write(f"{question['question_num']}:\n")
                        report.write(f"  Original timestamp: {question['original_timestamp']}\n")
                        report.write(f"  Corrected timestamp: {question['timestamp']}\n\n")
                    else:
                        report.write(f"{question['question_num']}: {question['timestamp']} (validated)\n\n")
                
                # Update question file with corrected timestamps
                self.update_question_file(question_file, content, corrected_questions)
                
                report.write("-" * 60 + "\n\n")
                
        print(f"Timestamp validation and correction complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate and correct timestamps in generated quiz questions')
    parser.add_argument('--file', type=str, required=True, help='Path to original transcript docx file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output', help='Directory containing generated question files')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', help='Claude model to use')
    parser.add_argument('--target-pages', type=int, default=35, 
                        help='Target number of pages per chunk (should match original processing)')
    
    args = parser.parse_args()
    
    corrector = TimestampCorrector(api_key=args.api_key, model=args.model)
    corrector.process_all_chunks(args.file, args.output_dir, args.target_pages)