# 🤖 AI-Powered Resume Job Matcher

An intelligent resume-job matching application built with Streamlit, LangChain, OpenAI, and ChromaDB that helps HR professionals and job seekers find the best matches using advanced vector similarity search and AI analysis.

## 🌟 Features

- **Smart Resume-Job Matching**: Find the best job opportunities for resumes using AI-powered similarity matching
- **Reverse Matching**: Find the most suitable candidates for job descriptions
- **Multi-format Support**: Process PDF, DOCX, and TXT resume files
- **Vector Search**: Advanced semantic search using OpenAI embeddings and ChromaDB
- **Detailed Analysis**: AI-powered match analysis with skill gaps and recommendations
- **Interactive Chat**: Conversational AI interface for matching insights
- **Analytics Dashboard**: Visual insights into matching performance
- **User-friendly Interface**: Clean Streamlit web application

## 🏗️ Architecture

The application consists of several key components:

- **`app.py`**: Main Streamlit application with user interface
- **`config.py`**: Configuration management and environment variables
- **`data_processor.py`**: Document processing for resumes and job descriptions
- **`vector_store.py`**: ChromaDB vector store for similarity search
- **`resume_matcher.py`**: Core matching logic with LLM analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**

2. **Install dependencies**
- pip install -r requirements.txt

3. **Set up environment variables**
Create a `.env` file in the project root:
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4o-mini
SIMILARITY_THRESHOLD=0.7
TOP_K_MATCHES=5


4. **Run the application**
streamlit run app.py

## 📋 Usage

### 1. Data Management
- Upload resume files (PDF, DOCX, TXT)
- Upload job descriptions as CSV file
- Process and store documents in vector database
  <img width="1854" height="936" alt="Screenshot 2025-08-02 155451" src="https://github.com/user-attachments/assets/34d9d400-2893-4551-9aaf-475d6081aed7" />

### 2. Resume → Jobs Matching
- Paste resume content or select processed resume
- Get ranked list of matching job opportunities
- View detailed analysis including:
  - Match percentage
  - Matching skills
  - Missing qualifications
  - Improvement recommendations

### 3. Job → Resumes Matching
- Input job description
- Find best candidate matches
- Analyze candidate strengths and development areas
<img width="1844" height="937" alt="Screenshot 2025-08-02 155753" src="https://github.com/user-attachments/assets/ae4fc438-139d-42ab-98ad-6cb84ce3f96b" />

### 4. Chat Interface
- Ask questions about matching results
- Get AI-powered insights and recommendations
- Interactive conversation about career advice
<img width="1856" height="931" alt="Screenshot 2025-08-02 165837" src="https://github.com/user-attachments/assets/6942a4a6-90c6-4e24-8206-52ce46f271d9" />

### 5. Analytics Dashboard
- View matching statistics
- Track performance metrics
- Analyze trends and patterns
<img width="1860" height="933" alt="Screenshot 2025-08-02 165915" src="https://github.com/user-attachments/assets/32309926-eb37-4208-a62c-20eadc135634" />


## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and document processing
- **OpenAI**: Embeddings and language model
- **ChromaDB**: Vector database for similarity search
- **Pandas**: Data manipulation
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing

### Vector Search
- Uses OpenAI's `text-embedding-ada-002` for document embeddings
- ChromaDB for efficient similarity search
- Cosine similarity for matching scores
- Configurable similarity thresholds

### AI Analysis
- GPT-4o-mini for detailed match analysis
- Structured JSON responses for consistent formatting
- Skill gap analysis and recommendations
- Conversational AI for user queries

## ⚙️ Configuration

Customize the application behavior through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `CHROMA_PERSIST_DIRECTORY` | Vector database storage path | `./chroma_db` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-ada-002` |
| `LLM_MODEL` | Language model for analysis | `gpt-4o-mini` |
| `SIMILARITY_THRESHOLD` | Minimum similarity for matches | `0.7` |
| `TOP_K_MATCHES` | Default number of results | `5` |

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is valid and has sufficient credits
   - Check the key is properly set in the sidebar or `.env` file

2. **ChromaDB Initialization Issues**
   - Verify the persist directory path exists and is writable
   - Clear the database directory if corrupted

3. **File Processing Errors**
   - Ensure uploaded files are not corrupted
   - Check file formats are supported (PDF, DOCX, TXT)

4. **Memory Issues**
   - Large documents may cause memory problems
   - Consider reducing chunk sizes or document lengths

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- OpenAI for providing powerful language models and embeddings
- LangChain community for excellent documentation and tools
- Streamlit team for the amazing web framework
- ChromaDB for efficient vector storage

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

Built using Python, Streamlit, and AI

