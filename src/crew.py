import os
import yaml
import re
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool
from src.tools.file_tools import ResumeFileProcessor, ResumeGenerator

class ResumeAnalyzerCrew:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        
        # API Key validation
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("⚠️  WARNING: No API key found. Using dynamic fallback mode.")
            self.llm = None
        else:
            try:
                # FIX: Use gemini/ prefix AND explicitly set api_version to 'v1' to avoid 404 NOT_FOUND on v1beta
                self.llm = LLM(
                    model="gemini/gemini-1.5-flash",
                    api_key=self.api_key,
                    temperature=0.1,
                    api_version="v1" 
                )
                print("✓ Gemini LLM initialized (Stable V1 API)")
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}")
                self.llm = None
        
        self.agents_config = self._load_yaml_config('agents.yaml')
        self.tasks_config = self._load_yaml_config('tasks.yaml')
        self.setup_tools()
        self.setup_agents()

    def _load_yaml_config(self, filename):
        config_path = self.base_dir / 'config' / filename
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return self._get_default_config(filename)
        except Exception:
            return self._get_default_config(filename)

    def _get_default_config(self, filename):
        if filename == 'agents.yaml':
            return {'resume_analyzer': {'role': 'Senior Resume Analyst', 'goal': 'Analyze resumes', 'backstory': 'Expert analyst'}}
        elif filename == 'tasks.yaml':
            return {'analyze_resume': {'description': 'Analyze {file_path}', 'expected_output': 'Report'}}
        return {}

    def setup_tools(self):
        try:
            self.file_processor = ResumeFileProcessor()
            self.resume_generator = ResumeGenerator()
            self.file_read_tool = FileReadTool()
        except Exception as e:
            print(f"⚠️  Tool initialization error: {e}")

    def setup_agents(self):
        if self.llm:
            try:
                self.analyzer_agent = Agent(
                    config=self.agents_config['resume_analyzer'],
                    llm=self.llm,
                    tools=[self.file_processor, self.file_read_tool],
                    verbose=True,
                    allow_delegation=False
                )
            except Exception as e:
                print(f"⚠️  Agent setup error: {e}")
                self.analyzer_agent = None
        else:
            self.analyzer_agent = None

    async def analyze_with_static_knowledge(self, file_path: str):
        return await self._run_analysis(file_path)

    async def analyze_with_web_search(self, file_path: str):
        return await self._run_analysis(file_path)

    async def _run_analysis(self, file_path: str):
        """Core analysis logic with robust error handling and direct text injection"""
        resume_text = ""
        try:
            # 1. Extract text manually to ensure the AI has context
            if self.file_processor:
                resume_text = self.file_processor.extract_text(file_path)
            
            if not resume_text or "Error" in resume_text:
                return self._create_dynamic_fallback_analysis(file_path, "")

            # 2. If no AI available, use dynamic fallback
            if not self.analyzer_agent:
                return self._create_dynamic_fallback_analysis(file_path, resume_text)
            
            # 3. Direct Task Injection: Feed text directly into the prompt to ensure unique results
            task_desc = f"""
            Analyze the following resume for an AI/Data Science role.
            
            RESUME CONTENT:
            \"\"\"
            {resume_text[:7000]}
            \"\"\"
            
            Provide your analysis with these specific headers:
            ATS SCORE: [Number 0-100]
            STRENGTHS: [Bullet points]
            IMPROVEMENTS: [Bullet points]
            MISSING KEYWORDS: [Comma-separated list]
            """
            
            task = Task(
                description=task_desc,
                expected_output="Detailed ATS report with score, strengths, improvements, and missing keywords.",
                agent=self.analyzer_agent
            )
            
            crew = Crew(agents=[self.analyzer_agent], tasks=[task], verbose=True)
            
            print("🤖 Running AI analysis (V1 API)...")
            result = str(crew.kickoff())
            
            return self._robust_parse(result, resume_text)
            
        except Exception as e:
            print(f"❌ AI Failed ({e}). Using Local Analysis Engine.")
            return self._create_dynamic_fallback_analysis(file_path, resume_text)

    def _create_dynamic_fallback_analysis(self, file_path: str, content: str):
        """High-accuracy text-based analysis engine (No AI required)"""
        print(f"📊 Analyzing content locally: {Path(file_path).name}")
        
        if not content:
            try:
                content = self.file_processor.extract_text(file_path)
            except:
                content = ""
        
        content_lower = content.lower()
        
        # 1. Expanded Keyword Database
        keywords_db = {
            'Languages': ['python', 'sql', ' r ', 'java', 'c++', 'scala'],
            'Machine Learning': ['regression', 'classification', 'clustering', 'random forest', 'xgboost', 'scikit-learn', 'nlp', 'computer vision', 'time series'],
            'Deep Learning': ['neural network', 'pytorch', 'tensorflow', 'keras', 'cnn', 'rnn', 'transformers', 'bert', 'llm', 'gan'],
            'Data Engineering': ['spark', 'hadoop', 'kafka', 'etl', 'airflow', 'docker', 'kubernetes', 'aws', 'azure', 'gcp'],
            'Analytics/Viz': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'excel', 'statistics']
        }
        
        found_keywords = []
        for cat_kws in keywords_db.values():
            found_keywords.extend([kw.strip() for kw in cat_kws if kw in content_lower])
        
        # 2. Section Detection
        sections = {
            'Experience': ['experience', 'work history', 'professional background'],
            'Education': ['education', 'academic', 'degree', 'university'],
            'Projects': ['projects', 'portfolio', 'github'],
            'Skills': ['skills', 'technologies', 'competencies']
        }
        found_sections = [s for s, aliases in sections.items() if any(a in content_lower for a in aliases)]
        
        # 3. Quantification Check (Metrics)
        has_metrics = bool(re.search(r'\d+%', content)) or bool(re.search(r'\$\d+', content)) or "improved" in content_lower or "reduced" in content_lower
        
        # 4. Score Calculation (0-100)
        score = (len(set(found_keywords)) / 22) * 55 # Max 55 pts for keywords
        score += (len(found_sections) / 4) * 20      # Max 20 pts for sections
        score += 15 if has_metrics else 0           # Max 15 pts for metrics
        score += 10 if '@' in content else 0        # Max 10 pts for contact info
        
        ats_score = int(min(score, 100))
        
        # 5. Dynamic Strengths & Improvements
        strengths = [f"Found {len(set(found_keywords))} technical keywords relevant to Data Science"]
        if has_metrics: strengths.append("Includes quantified achievements and metrics")
        if 'Projects' in found_sections: strengths.append("Contains a dedicated projects/portfolio section")
        
        improvements = []
        if len(found_keywords) < 12: improvements.append("Increase keyword density for ML frameworks (e.g. PyTorch, Scikit-learn)")
        if not has_metrics: improvements.append("Add specific metrics (%, $, time saved) to your experience to quantify impact")
        if 'Skills' not in found_sections: improvements.append("Add a clearly labeled 'Technical Skills' section for better ATS scanning")
        if 'Projects' not in found_sections: improvements.append("Include specific personal or academic projects to demonstrate practical skills")

        missing_keywords = [kw.strip() for kws in keywords_db.values() for kw in kws if kw not in content_lower]

        return {
            "ats_score": max(ats_score, 10),
            "strengths": strengths[:4],
            "improvements": improvements[:4],
            "missing_keywords": missing_keywords[:12],
            "raw_result": "Local Engine Analysis"
        }

    def _robust_parse(self, text, original_text):
        """Extracts data from AI output with flexible regex to handle various formatting"""
        print("🔍 Extracting AI Results...")
        
        # Flexible Score Extraction (matches "Score: 85", "ATS SCORE - 85", etc.)
        score = 75
        score_match = re.search(r'(?:ATS\s*)?SCORE\s*[:\-]?\s*(\d+)', text, re.I)
        if score_match: score = int(score_match.group(1))
        
        def extract_list_robust(header, next_headers):
            # Matches header followed by any bulleted list until the next recognized header
            pattern = rf'(?:\*\*|#)?\s*{header}\s*(?:\*\*|#)?\s*[:\-]?\s*(.*?)(?=(?:\*\*|#)?\s*(?:{"|".join(next_headers)})\b|$)'
            match = re.search(pattern, text, re.I | re.S)
            if not match: return []
            
            # Find all bullet points or numbered items
            items = re.findall(r'(?:^|\n)\s*[-*•\d.]+\s*(.+)', match.group(1))
            if not items:
                # Fallback: if no bullets, split by newlines
                items = [line.strip() for line in match.group(1).split('\n') if len(line.strip()) > 5]
            
            return [i.strip() for i in items if len(i.strip()) > 5]

        header_list = ["STRENGTHS", "IMPROVEMENTS", "MISSING KEYWORDS", "SCORE", "WEAKNESSES"]
        
        strengths = extract_list_robust("STRENGTHS", header_list[1:])
        # Check both "IMPROVEMENTS" and "WEAKNESSES" headers
        improvements = extract_list_robust("IMPROVEMENTS", header_list[2:])
        if not improvements:
            improvements = extract_list_robust("WEAKNESSES", header_list[2:])
        
        # Keywords
        kw_match = re.search(r'(?:MISSING\s*)?KEYWORDS\s*[:\-]?\s*(.*)', text, re.I | re.S)
        keywords = []
        if kw_match:
            # Capture only the first line/paragraph of keywords
            kw_text = kw_match.group(1).split('\n\n')[0]
            keywords = [k.strip(' -*•') for k in re.split(r'[,|\n]', kw_text) if len(k.strip()) > 1]

        return {
            "ats_score": min(max(score, 0), 100),
            "strengths": strengths[:5] if strengths else ["Technical skills alignment"],
            "improvements": improvements[:5] if improvements else ["Quantify achievements with metrics"],
            "missing_keywords": keywords[:12],
            "raw_result": text
        }

    async def generate_enhanced_resume(self, file_path: str, analysis_data: dict):
        try:
            output_dir = self.base_dir / "outputs"
            output_dir.mkdir(exist_ok=True)
            output_name = f"enhanced_{Path(file_path).name}.docx"
            output_path = output_dir / output_name
            
            content = f"RESUME ENHANCEMENT SUMMARY\nATS Score: {analysis_data['ats_score']}/100\n\n"
            content += "STRENGTHS:\n" + "\n".join([f"- {s}" for s in analysis_data['strengths']])
            content += "\n\nIMPROVEMENTS:\n" + "\n".join([f"- {i}" for i in analysis_data['improvements']])
            
            self.resume_generator.generate_docx(content, str(output_path))
            return output_name
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return None
