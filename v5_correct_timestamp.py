import os
import re
import anthropic
import argparse
import docx
from typing import List, Dict, Tuple, Optional, Any

class TimestampAdder:
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the timestamp adder with API key and model.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a docx file.
        
        Args:
            file_path: Path to the docx file
            
        Returns:
            Full transcript text
        """
        doc = docx.Document(file_path)
        full_text = []
        
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        return "\n".join(full_text)

    def get_question_files(self, output_dir: str) -> List[str]:
        """
        Find all question files in the output directory.
        
        Args:
            output_dir: Directory containing generated question files
            
        Returns:
            List of file paths
        """
        question_files = []
        
        # Find all chunk_*_questions.txt files
        for file in os.listdir(output_dir):
            if file.endswith('_questions.txt') or file == 'all_questions.txt':
                file_path = os.path.join(output_dir, file)
                question_files.append(file_path)
        
        return question_files

    def read_content(self, file_path: str) -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content of the file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_questions(self, content: str) -> List[Dict]:
        """
        Extract questions and answers from the content.
        
        Args:
            content: Content of the question file
            
        Returns:
            List of dictionaries containing questions and their content
        """
        # Looking for patterns like "Question 1" or "## Question 1"
        question_pattern = r'(?:^|\n)(?:#+\s*)?(?:Question|Q)\s+(\d+)[\s\S]*?(?=(?:^|\n)(?:#+\s*)?(?:Question|Q)\s+\d+|$)'
        
        questions = []
        matches = re.findall(question_pattern, content, re.MULTILINE)
        
        # If no matches found, try an alternative pattern
        if not matches:
            alternative_pattern = r'(?:^|\n)(?:#+\s*)?(?:Question|Q)\s+(\d+)[\s\S]*?(?=(?:^|\n)(?:#+\s*)?(?:Question|Q)\s+\d+|$)'
            content_blocks = re.split(r'=== CHUNK \d+ ===', content)
            for block in content_blocks:
                if block.strip():
                    matches = re.findall(alternative_pattern, block, re.MULTILINE)
                    if matches:
                        for match in matches:
                            # Find the full question content
                            question_text_match = re.search(f'(?:^|\n)(?:#+\\s*)?(?:Question|Q)\\s+{match}([\\s\\S]*?)(?=(?:^|\n)(?:#+\\s*)?(?:Question|Q)\\s+\\d+|$)', block, re.MULTILINE)
                            if question_text_match:
                                question_content = question_text_match.group(1).strip()
                                questions.append({
                                    "question_num": match,
                                    "content": question_content,
                                    "full_text": f"Question {match}\n{question_content}"
                                })
        else:
            # Process the matches when using the first pattern
            for match in matches:
                question_text_match = re.search(f'(?:^|\n)(?:#+\\s*)?(?:Question|Q)\\s+{match}([\\s\\S]*?)(?=(?:^|\n)(?:#+\\s*)?(?:Question|Q)\\s+\\d+|$)', content, re.MULTILINE)
                if question_text_match:
                    question_content = question_text_match.group(1).strip()
                    questions.append({
                        "question_num": match,
                        "content": question_content,
                        "full_text": f"Question {match}\n{question_content}"
                    })
        
        # If still no matches, try a more aggressive approach by splitting on headers
        if not questions:
            # Split the content on lines that look like question headers
            parts = re.split(r'(?:^|\n)((?:#+\s*)?(?:Question|Q)\s+\d+)', content, flags=re.MULTILINE)
            
            if len(parts) > 1:
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        question_header = parts[i].strip()
                        question_content = parts[i+1].strip()
                        
                        # Extract just the question number
                        number_match = re.search(r'(?:Question|Q)\s+(\d+)', question_header)
                        if number_match:
                            question_num = number_match.group(1)
                            questions.append({
                                "question_num": question_num,
                                "content": question_content,
                                "full_text": f"{question_header}\n{question_content}"
                            })
        
        print(f"Extracted {len(questions)} questions")
        return questions

    def extract_chunks_from_file(self, file_path: str) -> List[Dict]:
        """
        Extract chunks from a file that contains multiple chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dictionaries containing chunk number and content
        """
        content = self.read_content(file_path)
        
        # Split the content into chunks
        chunk_pattern = r'=== CHUNK (\d+) ===\s*([\s\S]*?)(?===== CHUNK \d+ ===|$)'
        
        chunks = []
        matches = re.findall(chunk_pattern, content)
        
        if not matches:
            # If no chunks found, treat the whole file as one chunk
            chunks.append({
                "chunk_num": "1",
                "content": content
            })
        else:
            for match in matches:
                chunk_num = match[0]
                chunk_content = match[1].strip()
                
                chunks.append({
                    "chunk_num": chunk_num,
                    "content": chunk_content
                })
        
        print(f"Extracted {len(chunks)} chunks")
        return chunks

    def find_timestamp_for_question(self, transcript: str, question_dict: Dict) -> str:
        """
        Find the timestamp in the transcript where the topic of the question is discussed.
        
        Args:
            transcript: Full transcript text
            question_dict: Question dictionary
            
        Returns:
            Timestamp or timestamp range
        """
        # Extract just the question text (before the options)
        question_text = question_dict["content"]
        
        # Remove any existing timestamps from the question text
        question_text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?', '', question_text)
        
        # Extract just the question (without options and explanation)
        clean_question = question_text
        
        # Try to extract the actual question part (before options)
        options_match = re.search(r'(A\)|A\.|A\s*\.)[\s\S]*', question_text)
        if options_match:
            clean_question = question_text[:options_match.start()].strip()
        
        # Prepare prompt for Claude to find the timestamp
        prompt = f"""You are an expert at finding exact timestamps in educational transcripts.

I need you to find where in this transcript the topic of the following question is discussed:

Question {question_dict["question_num"]}: {clean_question}

The full question with options and explanation is:

{question_dict["full_text"]}

Instructions:
1. Search the transcript carefully for the section that discusses this specific topic
2. Return ONLY the timestamp or timestamp range in format "HH:MM" or "HH:MM - HH:MM"
3. If multiple sections discuss this, focus on the most relevant one
4. If you can't find an exact timestamp, indicate that with "Timestamp not found"

DO NOT include any explanations in your response - just the timestamp.

Here is the transcript:
{transcript}
"""
        
        # Get timestamp from Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract timestamp from Claude's response
        timestamp = response.content[0].text.strip()
        
        # Validate the timestamp format
        timestamp_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)'
        single_timestamp_pattern = r'^(\d{1,2}:\d{2}(?::\d{2})?)$'
        
        timestamp_match = re.search(timestamp_pattern, timestamp)
        single_match = re.search(single_timestamp_pattern, timestamp)
        
        if timestamp_match or single_match:
            return timestamp
        elif "not found" in timestamp.lower() or "couldn't find" in timestamp.lower():
            return "Timestamp not found"
        else:
            # Try to extract any timestamp-like pattern from the response
            timestamp_extraction = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?)', timestamp)
            if timestamp_extraction:
                return timestamp_extraction.group(1)
            else:
                return "Timestamp not found"

    def add_timestamps_to_questions(self, content: str, questions: List[Dict], timestamps: List[str]) -> str:
        """
        Add timestamps to the questions in the content.
        
        Args:
            content: Original content
            questions: List of question dictionaries
            timestamps: List of timestamps for each question
            
        Returns:
            Updated content with timestamps added
        """
        updated_content = content
        
        for i, question in enumerate(questions):
            if i < len(timestamps) and timestamps[i] != "Timestamp not found":
                # Try to find the question header and add the timestamp
                question_pattern = f'((?:^|\\n)(?:#+\\s*)?(?:Question|Q)\\s+{question["question_num"]})(?!\\s*\\()'
                replacement = f'\\1 ({timestamps[i]})'
                
                # Attempt the replacement
                new_content = re.sub(question_pattern, replacement, updated_content, flags=re.MULTILINE)
                
                # Only update if the replacement actually changed something
                if new_content != updated_content:
                    updated_content = new_content
        
        return updated_content

    def process_file(self, transcript: str, question_file: str, output_dir: str):
        """
        Process a single question file.
        
        Args:
            transcript: Full transcript text
            question_file: Path to the question file
            output_dir: Directory to save the updated file
        """
        print(f"Processing {os.path.basename(question_file)}...")
        
        # Check if this is the all_questions.txt file
        is_all_questions = os.path.basename(question_file) == 'all_questions.txt'
        
        if is_all_questions:
            # Process each chunk separately
            chunks = self.extract_chunks_from_file(question_file)
            
            updated_content = ""
            
            for chunk in chunks:
                print(f"Processing Chunk {chunk['chunk_num']}...")
                
                # Extract questions from this chunk
                questions = self.extract_questions(chunk["content"])
                
                if not questions:
                    print(f"No questions found in Chunk {chunk['chunk_num']}")
                    updated_content += f"=== CHUNK {chunk['chunk_num']} ===\n\n{chunk['content']}\n\n"
                    continue
                
                # Find timestamps for each question
                timestamps = []
                for question in questions:
                    print(f"Finding timestamp for Chunk {chunk['chunk_num']} Question {question['question_num']}...")
                    timestamp = self.find_timestamp_for_question(transcript, question)
                    timestamps.append(timestamp)
                    print(f"  Chunk {chunk['chunk_num']} Question {question['question_num']}: {timestamp}")
                
                # Add timestamps to the chunk content
                chunk_updated = self.add_timestamps_to_questions(chunk["content"], questions, timestamps)
                
                # Add to the overall updated content
                updated_content += f"=== CHUNK {chunk['chunk_num']} ===\n\n{chunk_updated}\n\n"
                
                # Create a report for this chunk
                report_file = os.path.join(output_dir, f"chunk_{chunk['chunk_num']}_timestamp_report.txt")
                
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(f"# Timestamp Report for Chunk {chunk['chunk_num']}\n\n")
                    
                    for j, question in enumerate(questions):
                        if j < len(timestamps):
                            f.write(f"Question {question['question_num']}: {timestamps[j]}\n")
                            # Extract just the question without options
                            clean_question = question["content"]
                            options_match = re.search(r'(A\)|A\.|A\s*\.)[\s\S]*', clean_question)
                            if options_match:
                                clean_question = clean_question[:options_match.start()].strip()
                            f.write(f"Question text: {clean_question[:100]}...\n\n")
                
                print(f"Saved timestamp report for Chunk {chunk['chunk_num']} to {report_file}")
        else:
            # Process the file as a single chunk
            content = self.read_content(question_file)
            
            # Extract questions
            questions = self.extract_questions(content)
            
            if not questions:
                print(f"No questions found in {question_file}")
                return
            
            # Find timestamps for each question
            timestamps = []
            for question in questions:
                print(f"Finding timestamp for Question {question['question_num']}...")
                timestamp = self.find_timestamp_for_question(transcript, question)
                timestamps.append(timestamp)
                print(f"  Question {question['question_num']}: {timestamp}")
            
            # Add timestamps to the questions
            updated_content = self.add_timestamps_to_questions(content, questions, timestamps)
            
            # Create a report for this file
            base_name = os.path.basename(question_file)
            report_file = os.path.join(output_dir, base_name.replace('.txt', '_timestamp_report.txt'))
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(f"# Timestamp Report for {base_name}\n\n")
                
                for i, question in enumerate(questions):
                    if i < len(timestamps):
                        f.write(f"Question {question['question_num']}: {timestamps[i]}\n")
                        # Extract just the question without options
                        clean_question = question["content"]
                        options_match = re.search(r'(A\)|A\.|A\s*\.)[\s\S]*', clean_question)
                        if options_match:
                            clean_question = clean_question[:options_match.start()].strip()
                        f.write(f"Question text: {clean_question[:100]}...\n\n")
            
            print(f"Saved timestamp report to {report_file}")
        
        # Save the updated content
        timestamped_file = question_file.replace('.txt', '_timestamped.txt')
        
        with open(timestamped_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
            
        print(f"Saved timestamped version to {timestamped_file}")

    def process_all_files(self, transcript_file: str, output_dir: str):
        """
        Process all question files.
        
        Args:
            transcript_file: Path to the transcript file
            output_dir: Directory containing question files
        """
        # Extract transcript text
        print(f"Reading transcript from {transcript_file}...")
        transcript = self.extract_text_from_docx(transcript_file)
        
        # Get all question files
        question_files = self.get_question_files(output_dir)
        print(f"Found {len(question_files)} question files")
        
        # Process each file
        for file in question_files:
            self.process_file(transcript, file, output_dir)
            
        print(f"Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add timestamps to questions based on transcript content')
    parser.add_argument('--file', type=str, required=True, help='Path to transcript docx file')
    parser.add_argument('--api-key', type=str, required=True, help='Anthropic API key')
    parser.add_argument('--output-dir', type=str, default='quiz_output', help='Directory containing question files')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219', help='Claude model to use')
    
    args = parser.parse_args()
    
    adder = TimestampAdder(api_key=args.api_key, model=args.model)
    adder.process_all_files(args.file, args.output_dir)