# data_processor.py
import os
import re
from typing import List, Dict
import PyPDF2
from docx import Document
import pandas as pd

class DataProcessor:
    def __init__(self):
        self.resume_data = []
        self.job_data = []
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\@]', ' ', text)
        return text.strip()
    
    def process_resumes(self, resume_folder: str) -> List[Dict]:
        """Process all resume files in the folder"""
        resumes = []
        
        for filename in os.listdir(resume_folder):
            file_path = os.path.join(resume_folder, filename)
            
            if filename.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text = self.extract_text_from_docx(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                continue
            
            if text:
                resumes.append({
                    'id': filename,
                    'filename': filename,
                    'content': self.clean_text(text),
                    'type': 'resume'
                })
        
        return resumes
    
    def process_jobs(self, jobs_file: str) -> List[Dict]:
        """Process job descriptions from CSV file"""
        jobs = []
        
        try:
            df = pd.read_csv(jobs_file)
            for index, row in df.iterrows():
                job_desc = f"{row.get('title', '')} {row.get('description', '')} {row.get('requirements', '')}"
                jobs.append({
                    'id': f"job_{index}",
                    'title': row.get('title', 'Unknown'),
                    'company': row.get('company', 'Unknown'),
                    'content': self.clean_text(job_desc),
                    'type': 'job'
                })
        except Exception as e:
            print(f"Error processing jobs file: {e}")
        
        return jobs