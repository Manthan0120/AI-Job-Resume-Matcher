# resume_matcher.py
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import List, Dict, Tuple
import json

class ResumeJobMatcher:
    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            temperature=0.1,
            model_name=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.setup_prompts()
    
    def setup_prompts(self):
        """Setup prompt templates"""
        self.match_prompt = PromptTemplate(
            input_variables=["resume_content", "job_content", "similarity_score"],
            template="""
            You are an expert HR consultant analyzing resume-job matches.
            
            Resume Content:
            {resume_content}
            
            Job Description:
            {job_content}
            
            Similarity Score: {similarity_score}
            
            Please provide a detailed analysis including:
            1. Match percentage (0-100%)
            2. Key matching skills and experiences
            3. Missing qualifications or gaps
            4. Recommendations for improvement
            5. Overall assessment
            
            Format your response as JSON with the following structure:
            {{
                "match_percentage": <percentage>,
                "matching_skills": [<list of skills>],
                "missing_qualifications": [<list of gaps>],
                "recommendations": [<list of recommendations>],
                "overall_assessment": "<assessment>"
            }}
            """
        )
    
    def find_best_matches(self, resume_content: str, top_k: int = 5) -> List[Dict]:
        """Find best job matches for a resume"""
        # Search for similar job descriptions
        job_results = self.vector_store.similarity_search(
            resume_content, 
            doc_type="job", 
            k=top_k * 2  # Get more results for better filtering
        )
        
        matches = []
        processed_jobs = set()
        
        for doc, score in job_results:
            job_id = doc.metadata['id']
            
            # Avoid duplicate jobs
            if job_id in processed_jobs:
                continue
            processed_jobs.add(job_id)
            
            # Calculate detailed similarity
            similarity = self.vector_store.calculate_cosine_similarity(
                resume_content, 
                doc.page_content
            )
            
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                # Get detailed analysis from LLM
                analysis = self.analyze_match(
                    resume_content, 
                    doc.page_content, 
                    similarity
                )
                
                matches.append({
                    'job_id': job_id,
                    'title': doc.metadata.get('title', 'Unknown'),
                    'company': doc.metadata.get('company', 'Unknown'),
                    'similarity_score': similarity,
                    'vector_score': 1 - score,  # Convert distance to similarity
                    'analysis': analysis,
                    'content': doc.page_content
                })
        
        # Sort by similarity score and return top k
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_k]
    
    def find_best_resumes(self, job_content: str, top_k: int = 5) -> List[Dict]:
        """Find best resume matches for a job"""
        # Search for similar resumes
        resume_results = self.vector_store.similarity_search(
            job_content, 
            doc_type="resume", 
            k=top_k * 2
        )
        
        matches = []
        processed_resumes = set()
        
        for doc, score in resume_results:
            resume_id = doc.metadata['id']
            
            if resume_id in processed_resumes:
                continue
            processed_resumes.add(resume_id)
            
            similarity = self.vector_store.calculate_cosine_similarity(
                job_content, 
                doc.page_content
            )
            
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                analysis = self.analyze_match(
                    doc.page_content, 
                    job_content, 
                    similarity
                )
                
                matches.append({
                    'resume_id': resume_id,
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'similarity_score': similarity,
                    'vector_score': 1 - score,
                    'analysis': analysis,
                    'content': doc.page_content
                })
        
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_k]
    
    def analyze_match(self, resume_content: str, job_content: str, similarity_score: float) -> Dict:
        """Analyze match using LLM"""
        try:
            prompt = self.match_prompt.format(
                resume_content=resume_content[:2000],  # Limit content length
                job_content=job_content[:2000],
                similarity_score=f"{similarity_score:.2f}"
            )
            
            response = self.llm.predict(prompt)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis = {
                    "match_percentage": int(similarity_score * 100),
                    "matching_skills": ["Skills analysis failed"],
                    "missing_qualifications": ["Analysis failed"],
                    "recommendations": ["Please retry analysis"],
                    "overall_assessment": response[:500]
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return {
                "match_percentage": int(similarity_score * 100),
                "matching_skills": ["Analysis unavailable"],
                "missing_qualifications": ["Analysis unavailable"],
                "recommendations": ["Please retry"],
                "overall_assessment": "Analysis failed due to technical error"
            }
    
    def get_chat_response(self, question: str, context: str = "") -> str:
        """Get conversational response about matches"""
        try:
            prompt = f"""
            Context: {context}
            
            User Question: {question}
            
            Please provide a helpful response about resume-job matching based on the context provided.
            """
            
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    