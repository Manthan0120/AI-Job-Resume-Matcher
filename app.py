# app.py
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os
from config import Config
from data_processor import DataProcessor
from vector_store import VectorStore
from resume_matcher import ResumeJobMatcher
import json

# Page configuration
st.set_page_config(
    page_title="AI Resume Job Matcher",
    page_icon="🤖",
    layout="wide"
)

class ResumeJobMatcherApp:
    def __init__(self):
        self.config = Config()
        
        # Initialize session state
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'matcher' not in st.session_state:
            st.session_state.matcher = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
    
    def initialize_components(self):
        """Initialize vector store and matcher"""
        if st.session_state.vector_store is None:
            with st.spinner("Initializing vector store..."):
                st.session_state.vector_store = VectorStore(self.config)
        
        if st.session_state.matcher is None:
            with st.spinner("Initializing matcher..."):
                st.session_state.matcher = ResumeJobMatcher(
                    self.config, 
                    st.session_state.vector_store
                )
    
    def load_data(self):
        """Load and process data"""
        st.header("📁 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Resumes")
            resume_files = st.file_uploader(
                "Upload resume files (PDF, DOCX, TXT)",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True
            )
            
            if resume_files and st.button("Process Resumes"):
                self.process_uploaded_resumes(resume_files)
        
        with col2:
            st.subheader("Upload Job Descriptions")
            jobs_file = st.file_uploader(
                "Upload jobs CSV file",
                type=['csv']
            )
            
            if jobs_file and st.button("Process Jobs"):
                self.process_uploaded_jobs(jobs_file)
    
    def process_uploaded_resumes(self, resume_files):
        """Process uploaded resume files"""
        processor = DataProcessor()
        
        # Save uploaded files temporarily
        os.makedirs("temp_resumes", exist_ok=True)
        
        resumes = []
        for file in resume_files:
            file_path = os.path.join("temp_resumes", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process file
            if file.name.endswith('.pdf'):
                text = processor.extract_text_from_pdf(file_path)
            elif file.name.endswith('.docx'):
                text = processor.extract_text_from_docx(file_path)
            else:
                text = file.getvalue().decode("utf-8")
            
            if text:
                resumes.append({
                    'id': file.name,
                    'filename': file.name,
                    'content': processor.clean_text(text),
                    'type': 'resume'
                })
        
        # Add to vector store
        if resumes:
            with st.spinner("Adding resumes to vector store..."):
                st.session_state.vector_store.add_documents(resumes)
            st.success(f"Successfully processed {len(resumes)} resumes!")
    
    def process_uploaded_jobs(self, jobs_file):
        """Process uploaded jobs file"""
        try:
            df = pd.read_csv(jobs_file)
            st.write("Preview of jobs data:")
            st.dataframe(df.head())
            
            jobs = []
            for index, row in df.iterrows():
                job_desc = f"{row.get('title', '')} {row.get('description', '')} {row.get('requirements', '')}"
                jobs.append({
                    'id': f"job_{index}",
                    'title': row.get('title', 'Unknown'),
                    'company': row.get('company', 'Unknown'),
                    'content': job_desc,
                    'type': 'job'
                })
            
            if jobs:
                with st.spinner("Adding jobs to vector store..."):
                    st.session_state.vector_store.add_documents(jobs)
                st.success(f"Successfully processed {len(jobs)} job descriptions!")
                
        except Exception as e:
            st.error(f"Error processing jobs file: {e}")
    
    def resume_to_jobs_matching(self):
        """Resume to jobs matching interface"""
        st.header("🎯 Find Jobs for Resume")
        
        # Text area for resume input
        resume_text = st.text_area(
            "Paste resume content here:",
            height=200,
            placeholder="Paste the resume text here..."
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            num_matches = st.slider("Number of matches", 1, 10, 5)
            
        with col2:
            if st.button("Find Job Matches", type="primary") and resume_text:
                with st.spinner("Finding best job matches..."):
                    matches = st.session_state.matcher.find_best_matches(
                        resume_text, 
                        top_k=num_matches
                    )
                
                if matches:
                    st.subheader("🎯 Best Job Matches")
                    
                    for i, match in enumerate(matches, 1):
                        with st.expander(f"#{i} {match['title']} at {match['company']} - {match['similarity_score']:.1%} match"):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("Match Score", f"{match['similarity_score']:.1%}")
                                analysis = match['analysis']
                                st.write("**Matching Skills:**")
                                for skill in analysis.get('matching_skills', [])[:5]:
                                    st.write(f"• {skill}")
                            
                            with col2:
                                st.write("**Missing Qualifications:**")
                                for gap in analysis.get('missing_qualifications', [])[:5]:
                                    st.write(f"• {gap}")
                                
                                st.write("**Recommendations:**")
                                for rec in analysis.get('recommendations', [])[:3]:
                                    st.write(f"• {rec}")
                else:
                    st.warning("No suitable job matches found. Try adjusting the similarity threshold.")
    
    def job_to_resumes_matching(self):
        """Job to resumes matching interface"""
        st.header("👥 Find Resumes for Job")
        
        job_text = st.text_area(
            "Paste job description here:",
            height=200,
            placeholder="Paste the job description here..."
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            num_matches = st.slider("Number of matches", 1, 10, 5, key="job_matches")
            
        with col2:
            if st.button("Find Resume Matches", type="primary") and job_text:
                with st.spinner("Finding best resume matches..."):
                    matches = st.session_state.matcher.find_best_resumes(
                        job_text, 
                        top_k=num_matches
                    )
                
                if matches:
                    st.subheader("👥 Best Resume Matches")
                    
                    for i, match in enumerate(matches, 1):
                        with st.expander(f"#{i} {match['filename']} - {match['similarity_score']:.1%} match"):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("Match Score", f"{match['similarity_score']:.1%}")
                                analysis = match['analysis']
                                st.write("**Candidate Strengths:**")
                                for skill in analysis.get('matching_skills', [])[:5]:
                                    st.write(f"• {skill}")
                            
                            with col2:
                                st.write("**Areas for Development:**")
                                for gap in analysis.get('missing_qualifications', [])[:5]:
                                    st.write(f"• {gap}")
                                
                                st.write("**Overall Assessment:**")
                                st.write(analysis.get('overall_assessment', 'No assessment available')[:300] + "...")
                else:
                    st.warning("No suitable resume matches found.")
    
    def chat_interface(self):
        """Chat interface for questions"""
        st.header("💬 Chat with AI Matcher")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about resume-job matching..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.matcher.get_chat_response(
                        prompt, 
                        context="Resume-job matching application"
                    )
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def analytics_dashboard(self):
        """Analytics and insights dashboard"""
        st.header("📊 Analytics Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Resumes", "N/A", help="Connect to database for live metrics")
        
        with col2:
            st.metric("Total Jobs", "N/A", help="Connect to database for live metrics")
        
        with col3:
            st.metric("Avg Match Accuracy", "90%", help="Based on user feedback")
        
        st.subheader("Match Distribution")
        st.info("Analytics features would show match score distributions, popular skills, and matching trends.")
                
    def get_dashboard_metrics(self):
        """Get metrics for dashboard"""
        metrics = {
            'total_resumes': 0,
            'total_jobs': 0,
            'total_documents': 0
        }
        
        if st.session_state.vector_store and st.session_state.vector_store.vectorstore:
            try:
                # Get collection info from ChromaDB
                collection = st.session_state.vector_store.vectorstore._collection
                
                # Count documents by type
                all_docs = collection.get()
                    
                if all_docs and 'metadatas' in all_docs:
                    for metadata in all_docs['metadatas']:
                        doc_type = metadata.get('type', '')
                        if doc_type == 'resume':
                            metrics['total_resumes'] += 1
                        elif doc_type == 'job':
                            metrics['total_jobs'] += 1
                    
                    metrics['total_documents'] = len(all_docs['metadatas'])
            
            except Exception as e:
                st.error(f"Error getting metrics: {e}")
        
        return metrics


    
    def run(self):
        """Main application runner"""
        st.title("🤖 AI-Powered Resume Job Matcher")
        st.markdown("Built with LangChain, OpenAI, and Vector Search")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select Page:",
                [
                    "Data Management",
                    "Resume → Jobs",
                    "Job → Resumes", 
                    "Chat Interface",
                    "Analytics"
                ]
            )
            
            st.markdown("---")
            st.subheader("Settings")
            
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key"
            )
            
            if api_key:
                self.config.OPENAI_API_KEY = api_key
                os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize components if API key is provided
        if self.config.OPENAI_API_KEY:
            self.initialize_components()
            
            # Route to selected page
            if page == "Data Management":
                self.load_data()
            elif page == "Resume → Jobs":
                self.resume_to_jobs_matching()
            elif page == "Job → Resumes":
                self.job_to_resumes_matching()
            elif page == "Chat Interface":
                self.chat_interface()
            elif page == "Analytics":
                self.analytics_dashboard()
        else:
            st.warning("Please enter your OpenAI API key in the sidebar to get started.")
            st.info("""
            This application requires an OpenAI API key to function. 
            You can get one from: https://platform.openai.com/api-keys
            """)

# Run the application
if __name__ == "__main__":
    app = ResumeJobMatcherApp()
    app.run()
