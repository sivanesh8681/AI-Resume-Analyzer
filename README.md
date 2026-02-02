# AI Resume Analyzer & Enhancer

🚀 An intelligent web application that analyzes resumes for AI/Data Science roles, provides actionable improvement suggestions, and generates ATS-optimized enhanced resumes using advanced AI agents.

## 🌟 Features

- **Intelligent Resume Analysis**: Comprehensive analysis of resumes against AI/Data Science job requirements
- **ATS Compatibility Scoring**: Automated scoring for Applicant Tracking System compatibility
- **Keyword Optimization**: Identifies missing keywords and suggests natural integration
- **Content Enhancement**: AI-powered suggestions for improving resume content and structure
- **Professional Resume Generation**: Creates polished, ATS-friendly DOCX resumes
- **Web Research Integration**: Real-time job market analysis for current requirements
- **Modern Web Interface**: Clean, responsive UI with drag-and-drop file upload
- **Multi-Modal Analysis**: Choose between quick static analysis or web-enhanced analysis

## 🏗️ Architecture

### Core Components

- **FastAPI Backend**: High-performance async web framework
- **CrewAI Framework**: Multi-agent AI system with specialized roles
- **React-like Frontend**: Modern HTML/CSS/JavaScript interface
- **Document Processing**: PDF and DOCX file handling with python-docx and PyPDF2

### AI Agents

1. **Resume Analyzer**: Analyzes technical skills, ATS compatibility, and content quality
2. **Resume Enhancer**: Provides specific improvement suggestions and keyword additions
3. **Resume Generator**: Creates professional, ATS-optimized resume documents
4. **Web Researcher**: Searches current job postings for market requirements

### Tools & Utilities

- **File Processing**: Extract text from PDF/DOCX files
- **Content Analysis**: Comprehensive resume structure and content evaluation
- **Web Search**: Job market research and requirements gathering
- **Document Generation**: Professional DOCX resume creation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Resume_Analyzer-main
   ```

2. **Run the setup script**
   ```bash
   python run.py
   ```
   This will:
   - Check and install required dependencies
   - Create necessary directories
   - Set up configuration files

3. **Configure API Keys** (Optional for enhanced features)
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Start the application**
   ```bash
   python run.py
   ```
   The application will be available at `http://localhost:8000`

## 📋 Usage

1. **Access the Web Interface**
   - Open `http://localhost:8000` in your browser
   - Upload your resume (PDF or DOCX format)

2. **Choose Analysis Mode**
   - **Quick Analysis**: Fast analysis using built-in knowledge base
   - **Web-Enhanced**: Real-time job market research (slower but more comprehensive)

3. **Review Analysis Results**
   - ATS compatibility score
   - Strengths and areas for improvement
   - Missing keywords and suggestions

4. **Generate Enhanced Resume**
   - Click "Generate Enhanced Resume"
   - Download the optimized DOCX file

## 🔧 Configuration

### Agents Configuration (`config/agents.yaml`)

Customize AI agent behaviors:
```yaml
resume_analyzer:
  role: "Senior Resume Analyst"
  goal: "Analyze resumes against job requirements..."
  verbose: true
  max_iter: 15
```

### Tasks Configuration (`config/tasks.yaml`)

Define agent tasks and expected outputs:
```yaml
analyze_resume:
  description: "Analyze the uploaded resume..."
  expected_output: "A comprehensive analysis report..."
```

## 📊 Analysis Features

### ATS Compatibility Analysis
- Email and phone number detection
- Keyword density evaluation
- File format optimization
- Section structure validation

### Content Quality Assessment
- Action verb usage
- Quantified achievements
- Professional online presence
- Technical skills alignment

### Keyword Optimization
- Missing skill identification
- Natural keyword integration
- Industry-specific terminology
- ATS-friendly keyword density

## 🛠️ Development

### Project Structure
```
Resume_Analyzer-main/
├── main.py                 # FastAPI application
├── run.py                  # Setup and launcher script
├── test_resume_generation.py  # Testing utilities
├── requirements.txt        # Python dependencies
├── config/
│   ├── agents.yaml        # AI agent configurations
│   └── tasks.yaml         # Task definitions
├── src/
│   ├── __init__.py
│   ├── crew.py            # CrewAI implementation
│   ├── knowledge/
│   │   └── ai_ds_requirements.py  # Domain knowledge base
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── analysis_tools.py     # Resume analysis tools
│   │   ├── file_tools.py         # File processing tools
│   │   └── web_tools.py          # Web search tools
│   └── ui/
│       ├── static/
│       │   ├── style.css
│       │   └── templates/
│       └── templates/
│           └── index.html        # Web interface
├── uploads/               # Uploaded resume files
├── outputs/               # Generated enhanced resumes
└── .gitignore
```

### Testing

Run the test suite:
```bash
python test_resume_generation.py
```

This tests:
- Basic resume generation
- Content enhancement
- File parsing functionality

### API Endpoints

- `GET /` - Web interface
- `POST /upload-resume` - Upload and analyze resume
- `POST /generate-resume` - Generate enhanced resume
- `GET /download/{filename}` - Download generated resume
- `GET /health` - Health check
- `POST /cleanup` - Clean up files (development)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CrewAI**: Multi-agent AI framework
- **FastAPI**: Modern Python web framework
- **Gemini/OpenAI**: AI language models
- **python-docx**: Document generation
- **PyPDF2**: PDF text extraction

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **API Key Issues**
   - For web-enhanced analysis, configure API keys in `.env`
   - Without API keys, the app falls back to static analysis

3. **File Upload Issues**
   - Check file size limits (10MB max)
   - Ensure PDF/DOCX format

4. **Generation Failures**
   - Verify write permissions in `outputs/` directory
   - Check disk space availability

### Debug Mode

Run with debug logging:
```bash
uvicorn main:app --reload --log-level debug
```