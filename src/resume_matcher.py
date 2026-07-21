# resume_matcher.py
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Tuple
import json

class ResumeJobMatcher:
    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY
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
    
    def get_chat_response(self, question: str, chat_history: List = []) -> Dict:
        """Get conversational response with context retrieval"""
        try:
            # Retrieve relevant documents from vector store
            relevant_docs = self.vector_store.similarity_search(
                query=question,
                k=5  # Get top 5 relevant chunks
            )
            
            # Build context from retrieved documents
            context_parts = []
            for doc, score in relevant_docs:
                doc_type = doc.metadata.get('type', 'unknown')
                if doc_type == 'resume':
                    context_parts.append(f"Resume ({doc.metadata.get('filename')}): {doc.page_content}")
                elif doc_type == 'job':
                    context_parts.append(f"Job ({doc.metadata.get('title')}): {doc.page_content}")
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
            
            # Format chat history for context
            history_text = ""
            for msg in chat_history[-3:]:  # Last 3 exchanges
                history_text += f"{msg['role']}: {msg['content']}\n"
            
            # Create enhanced prompt with retrieved context
            prompt = f"""You are an AI assistant for a resume-job matching application. 
    Use the following context from the database to answer the user's question accurately.
    
    Retrieved Context:
    {context}
    
    Chat History:
    {history_text}
    
    User Question: {question}
    
    Provide a helpful, accurate response based on the retrieved information. If the context doesn't contain relevant information, say so."""
    
            response = self.llm.predict(prompt)
            
            return {
                "answer": response,
                "source_documents": [doc.metadata for doc, _ in relevant_docs],
                "context_used": len(relevant_docs) > 0
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "context_used": False
            }
    
        