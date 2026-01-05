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
            
            # Display CSV preview and column names
            st.write("**Preview of jobs data:**")
            st.dataframe(df.head())
            st.write(f"**Columns found:** {', '.join(df.columns.tolist())}")
            
            jobs = []
            skipped = 0
            
            for index, row in df.iterrows():
                # Try to extract job content from various possible column names
                title = (row.get('title') or row.get('job_title') or 
                        row.get('Title') or row.get('Job Title') or 
                        row.get('position') or 'Unknown Position')
                
                description = (row.get('description') or row.get('job_description') or 
                              row.get('Description') or row.get('Job Description') or 
                              row.get('details') or '')
                
                requirements = (row.get('requirements') or row.get('qualifications') or 
                               row.get('Requirements') or row.get('Qualifications') or 
                               row.get('skills') or row.get('Skills') or '')
                
                company = (row.get('company') or row.get('Company') or 
                          row.get('employer') or row.get('Employer') or 'Unknown Company')
                
                # Combine all text fields
                job_desc = f"Title: {title}\n\nDescription: {description}\n\nRequirements: {requirements}"
                
                # Validate that content is not empty
                if job_desc.strip() and len(job_desc.strip()) > 20:
                    jobs.append({
                        'id': f"job_{index}",
                        'title': str(title),
                        'company': str(company),
                        'content': job_desc.strip(),
                        'type': 'job'
                    })
                else:
                    skipped += 1
                    st.warning(f"⚠️ Skipped row {index + 1}: Insufficient content")
            
            # Display processing results
            st.write(f"**Processed:** {len(jobs)} jobs")
            if skipped > 0:
                st.warning(f"**Skipped:** {skipped} rows due to insufficient content")
            
            # Show sample of what will be added
            if jobs:
                with st.expander("Preview first job content"):
                    st.text(jobs[0]['content'][:500] + "...")
            
            # Add to vector store
            if jobs:
                with st.spinner(f"Adding {len(jobs)} jobs to vector store..."):
                    st.session_state.vector_store.add_documents(jobs)
                st.success(f"✅ Successfully processed {len(jobs)} job descriptions!")
            else:
                st.error("❌ No valid jobs found in CSV. Please check your file format and column names.")
                st.info("""
                Expected column names (case-insensitive):
                - **title** or **job_title** or **position**
                - **description** or **job_description** or **details**
                - **requirements** or **qualifications** or **skills**
                - **company** or **employer** (optional)
                """)
                
        except Exception as e:
            st.error(f"❌ Error processing jobs file: {e}")
            st.exception(e)

    
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
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in message["sources"]:
                            st.write(f"• {src.get('type')}: {src.get('filename') or src.get('title')}")
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about resume-job matching..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response with context retrieval
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    result = st.session_state.matcher.get_chat_response(
                        prompt,
                        chat_history=st.session_state.messages
                    )
                    
                    response = result["answer"]
                    st.markdown(response)
                    
                    # Show sources if context was used
                    if result["context_used"]:
                        with st.expander("📚 Sources"):
                            for src in result["source_documents"]:
                                st.write(f"• {src.get('type')}: {src.get('filename') or src.get('title')}")
            
            # Add assistant response with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": result.get("source_documents", [])
            })

    
    def analytics_dashboard(self):
        """Analytics and insights dashboard"""
        st.header("📊 Analytics Dashboard")
        
        # Get metrics from vector store
        metrics = self.get_dashboard_metrics()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Resumes", 
                metrics['total_resumes'],
                help="Number of resumes in the database"
            )
        
        with col2:
            st.metric(
                "Total Jobs", 
                metrics['total_jobs'],
                help="Number of job descriptions in the database"
            )
        
        with col3:
            st.metric(
                "Total Documents", 
                metrics['total_documents'],
                help="Total chunks stored in vector database"
            )
        
        # Show data status
        if metrics['total_documents'] == 0:
            st.warning("⚠️ No data loaded yet. Please upload resumes and job descriptions in the Data Management section.")
        else:
            st.success(f"✅ Vector store contains {metrics['total_documents']} document chunks")
        
        # Additional metrics
        st.subheader("📈 Database Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics['total_resumes'] > 0:
                avg_chunks_per_resume = metrics['total_documents'] / max(metrics['total_resumes'], 1)
                st.metric("Avg Chunks per Resume", f"{avg_chunks_per_resume:.1f}")
        
        with col2:
            if metrics['total_jobs'] > 0:
                avg_chunks_per_job = metrics['total_documents'] / max(metrics['total_jobs'], 1)
                st.metric("Avg Chunks per Job", f"{avg_chunks_per_job:.1f}")
        
        st.subheader("Match Distribution")
        st.info("Analytics features would show match score distributions, popular skills, and matching trends.")

                
    def get_dashboard_metrics(self):
        """Get metrics for dashboard"""
        metrics = {
            'total_resumes': 0,
            'total_jobs': 0,
            'total_documents': 0
        }
        
        if not (st.session_state.vector_store and 
                st.session_state.vector_store.vectorstore):
            return metrics
        
        try:
            # Get all documents from ChromaDB
            all_docs = st.session_state.vector_store.vectorstore.get(
                include=['metadatas']
            )
            
            if all_docs and 'metadatas' in all_docs:
                metadatas = all_docs['metadatas']
                metrics['total_documents'] = len(metadatas)
                
                # Count UNIQUE resumes and jobs by their IDs
                unique_resume_ids = set()
                unique_job_ids = set()
                
                for metadata in metadatas:
                    doc_type = metadata.get('type', '')
                    doc_id = metadata.get('id', '')
                    
                    if doc_id:  # Only count if ID exists
                        if doc_type == 'resume':
                            unique_resume_ids.add(doc_id)
                        elif doc_type == 'job':
                            unique_job_ids.add(doc_id)
                
                metrics['total_resumes'] = len(unique_resume_ids)
                metrics['total_jobs'] = len(unique_job_ids)
                
                # Debug output
                st.sidebar.write(f"Debug: {metrics['total_documents']} chunks, "
                               f"{len(unique_resume_ids)} unique resumes, "
                               f"{len(unique_job_ids)} unique jobs")
                
        except Exception as e:
            st.error(f"⚠️ Error getting metrics: {str(e)}")
        
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