import re
from typing import Any, Optional, Type, List, Dict
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class AnalysisInput(BaseModel):
    """Input schema for resume analysis"""
    resume_text: str = Field(description="Resume text content to analyze")
    job_requirements: str = Field(default="", description="Job requirements or knowledge base to compare against")

class ResumeAnalyzer(BaseTool):
    name: str = "Resume Analyzer"
    description: str = "Analyze resume content for ATS compatibility, keyword matching, and structural issues"
    args_schema: Type[BaseModel] = AnalysisInput

    def _run(self, resume_text: str, job_requirements: str = "") -> str:
        """Perform comprehensive resume analysis"""
        try:
            # DEBUG: Print to verify resume text is changing
            print(f"\n{'='*60}")
            print(f"DEBUG: Analyzing resume with {len(resume_text)} characters")
            print(f"DEBUG: First 100 chars: {resume_text[:100]}")
            print(f"{'='*60}\n")
            
            # Validate inputs
            if not resume_text or len(resume_text.strip()) < 50:
                return "Error: Resume text is too short or empty for meaningful analysis."
            
            # If no job requirements provided, use default AI/DS requirements
            if not job_requirements:
                job_requirements = self._get_default_requirements()
            
            analysis_result = {
                "ats_score": self._calculate_ats_score(resume_text, job_requirements),
                "keyword_analysis": self._analyze_keywords(resume_text, job_requirements),
                "structure_analysis": self._analyze_structure(resume_text),
                "content_analysis": self._analyze_content(resume_text),
                "formatting_issues": self._check_formatting_issues(resume_text),
                "strengths": self._identify_strengths(resume_text, job_requirements),
                "weaknesses": self._identify_weaknesses(resume_text, job_requirements)
            }
            
            return self._format_analysis_report(analysis_result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in _run: {error_details}")
            return f"Error during analysis: {str(e)}. Please check the resume content and try again."
    
    def _get_default_requirements(self) -> str:
        """Default AI/DS job requirements if none provided"""
        return """
        Required Skills: Python, SQL, Machine Learning, Statistics, Data Analysis, Pandas, NumPy, 
        Scikit-learn, Git, Jupyter Notebooks, Data Visualization, Problem Solving.
        Preferred: TensorFlow, PyTorch, AWS, Docker, Deep Learning, NLP, Computer Vision, 
        Tableau, Power BI, Apache Spark, MongoDB, PostgreSQL.
        """
    
    def _calculate_ats_score(self, resume_text: str, job_requirements: str = "") -> int:
        """Calculate ATS compatibility score (0-10) using resume-derived signals
        
        FIXED: Now accepts job_requirements as parameter to use in keyword matching
        """
        try:
            score = 0.0
            text = resume_text or ""
            lower = text.lower()

            # Contact info (up to 2.5)
            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text):
                score += 1.5
            if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text):
                score += 1.0

            # Sections presence (up to 3.5)
            if re.search(r'\b(experience|work|employment|career|professional)\b', lower):
                score += 1.4
            if re.search(r'\b(skills|technical|competencies|technologies)\b', lower):
                score += 1.0
            if re.search(r'\b(education|degree|university|college|certificat)', lower):
                score += 0.8
            if re.search(r'\b(summary|objective|profile|about)\b', lower):
                score += 0.3

            # Quantified achievements & action verbs (up to 1.5)
            nums = re.findall(r'\b\d+\.?\d*%?\b', text)
            if len(nums) >= 3:
                score += 0.8
            action_verbs = ['developed','implemented','created','designed','built','led','managed','improved','optimized','achieved','delivered','analyzed','automated']
            action_count = sum(1 for v in action_verbs if v in lower)
            score += min(0.7, action_count * 0.1)

            # Word count heuristic (up to 1)
            wc = len(text.split())
            if 250 <= wc <= 900:
                score += 1.0
            elif wc < 200:
                score -= 0.5
            elif wc > 1200:
                score -= 0.5

            # Formatting penalties (deduct)
            formatting_issues = self._check_formatting_issues(text)
            score -= min(1.5, 0.4 * len(formatting_issues))

            # Keyword match contribution (up to 2.0)
            # FIXED: Properly use job_requirements parameter
            try:
                # Get keywords from job requirements (or default if not provided)
                if job_requirements and job_requirements.strip():
                    req_keywords = self._extract_tech_keywords(job_requirements)
                else:
                    req_keywords = self._extract_tech_keywords(self._get_default_requirements())
                
                # Also extract keywords from resume to make it dynamic
                resume_keywords = self._extract_tech_keywords(resume_text)
                
                # Combine both lists (deduplicated)
                keyword_universe = list(dict.fromkeys(req_keywords + resume_keywords))
                
                # Count how many keywords are found in resume
                found = sum(1 for k in keyword_universe if k.lower() in lower)
                
                if keyword_universe:
                    kw_ratio = found / max(1, len(keyword_universe))
                    score += min(2.0, kw_ratio * 2.0)
            except Exception as e:
                print(f"Warning: Keyword scoring failed: {e}")
                pass

            # Clamp and return integer 0-10
            final = max(0, min(10, round(score)))
            return int(final)

        except Exception as e:
            print(f"ERROR in _calculate_ats_score: {e}")
            return 5
    
    def _analyze_keywords(self, resume_text: str, job_requirements: str) -> Dict:
        """Analyze keyword matching between resume and requirements (uses both req and resume to drive results)"""
        try:
            resume_lower = (resume_text or "").lower()

            # Keywords from job requirements and from resume to build a dynamic universe
            req_keywords = self._extract_tech_keywords(job_requirements)
            resume_keywords = self._extract_tech_keywords(resume_text)

            # Preserve order and dedupe: requirements preferred, then resume-discovered
            keyword_universe = []
            seen = set()
            for k in (req_keywords + resume_keywords):
                key = k.lower()
                if key not in seen:
                    seen.add(key)
                    keyword_universe.append(k)

            found_keywords = [k for k in keyword_universe if k.lower() in resume_lower]
            missing_keywords = [k for k in keyword_universe if k.lower() not in resume_lower]

            total = len(keyword_universe)
            keyword_density = (len(found_keywords) / total) if total > 0 else 0.0

            return {
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords[:15],
                "keyword_density": round(keyword_density * 100, 1),
                "total_keywords_checked": total
            }
        except Exception as e:
            print(f"ERROR in _analyze_keywords: {e}")
            return {
                "found_keywords": [],
                "missing_keywords": [],
                "keyword_density": 0,
                "total_keywords_checked": 0
            }
    
    def _extract_tech_keywords(self, job_requirements: str) -> List[str]:
        """Extract technical keywords from provided text (requirements or resume). Returns deduped list preserving order."""
        text = (job_requirements or "").strip()
        ai_ds_keywords = [
            'Python','R','SQL','Java','Scala','JavaScript','Julia','Go',
            'Machine Learning','Deep Learning','Neural Networks','TensorFlow','PyTorch','Keras','Scikit-learn','XGBoost','Random Forest',
            'SVM','Regression','Classification','Clustering','NLP','Computer Vision','Reinforcement Learning',
            'Pandas','NumPy','Matplotlib','Seaborn','Plotly','Bokeh','Jupyter','Anaconda',
            'Statistics','Probability','Linear Algebra','Calculus','A/B Testing',
            'MySQL','PostgreSQL','MongoDB','Redis','SQLite','Oracle',
            'Apache Spark','Hadoop','Kafka',
            'AWS','Azure','GCP','S3','EC2','Lambda','BigQuery','Redshift','Snowflake',
            'Docker','Kubernetes','Git','GitHub','Jenkins','Airflow','MLflow',
            'Tableau','Power BI','Django','Flask','FastAPI','Streamlit','Excel','MATLAB'
        ]

        found = []
        lower = text.lower()

        # Prefer known tokens first
        for kw in ai_ds_keywords:
            if kw.lower() in lower and kw not in found:
                found.append(kw)

        # Extract technical-like tokens from the provided text (preserve order)
        custom_tokens = re.findall(r'\b[A-Za-z0-9+#\-\._]{2,}\b', text)
        for tok in custom_tokens:
            t = tok.strip().strip('.,;:()')
            if len(found) >= 80:
                break
            if len(t) > 2 and t.lower() not in [x.lower() for x in found] and t.lower() not in ['the','and','for','with','this','that','from','while','that']:
                found.append(t)

        # If nothing found, provide a sensible common fallback
        if not found:
            found = ['Python','SQL','Machine Learning','Statistics','Pandas','NumPy','Git','Data Analysis','Visualization','Excel']

        # Deduplicate while preserving order (case-insensitive)
        seen = set()
        dedup = []
        for k in found:
            key = k.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(k)

        return dedup
    
    def _analyze_structure(self, resume_text: str) -> Dict:
        """Analyze resume structure and organization"""
        try:
            lines = resume_text.split('\n')
            sections_found = []
            
            # Common section headers with more flexible matching
            section_patterns = {
                'contact': r'(contact|phone|email|address|linkedin|github)',
                'summary': r'(summary|objective|profile|about)',
                'experience': r'(experience|work|employment|career|professional)',
                'education': r'(education|qualification|degree|academic)',
                'skills': r'(skills|technical|competencies|technologies)',
                'projects': r'(projects|portfolio|work samples)',
                'certifications': r'(certifications?|certificates?|licensed?)'
            }
            
            for line in lines:
                line_clean = line.lower().strip()
                # Only consider lines that could be headers (not too long)
                if len(line.strip()) < 60 and len(line_clean) > 2:
                    for section, pattern in section_patterns.items():
                        if re.search(pattern, line_clean):
                            sections_found.append(section)
                            break
            
            # Also check for implicit contact info
            has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
            has_phone = bool(re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text))
            
            return {
                "sections_found": list(set(sections_found)),
                "total_sections": len(set(sections_found)),
                "has_contact": 'contact' in sections_found or has_email or has_phone,
                "has_summary": 'summary' in sections_found,
                "has_experience": 'experience' in sections_found,
                "has_skills": 'skills' in sections_found,
                "has_education": 'education' in sections_found,
                "word_count": len(resume_text.split()),
                "line_count": len([line for line in lines if line.strip()])
            }
        except Exception:
            return {
                "sections_found": [],
                "total_sections": 0,
                "has_contact": False,
                "has_summary": False,
                "has_experience": False,
                "has_skills": False,
                "has_education": False,
                "word_count": 0,
                "line_count": 0
            }
    
    def _analyze_content(self, resume_text: str) -> Dict:
        """Analyze content quality and completeness"""
        try:
            # Check for quantified achievements (numbers, percentages, etc.)
            numbers = re.findall(r'\b\d+\.?\d*%?\b', resume_text)
            percentages = re.findall(r'\b\d+\.?\d*%', resume_text)
            
            # Enhanced action verbs list
            action_verbs = [
                'developed', 'implemented', 'created', 'designed', 'built', 'led',
                'managed', 'improved', 'optimized', 'achieved', 'delivered',
                'analyzed', 'researched', 'collaborated', 'presented', 'deployed',
                'automated', 'streamlined', 'enhanced', 'established', 'coordinated',
                'executed', 'maintained', 'supervised', 'trained', 'mentored'
            ]
            
            found_verbs = []
            resume_lower = resume_text.lower()
            for verb in action_verbs:
                if verb in resume_lower:
                    found_verbs.append(verb)
            
            # Check for professional online presence
            has_github = bool(re.search(r'github\.com|github|git repository', resume_lower))
            has_linkedin = bool(re.search(r'linkedin\.com|linkedin', resume_lower))
            has_portfolio = any(word in resume_lower for word in ['portfolio', 'website', 'blog', 'personal site'])
            
            return {
                "quantified_achievements": len(numbers),
                "percentages_used": len(percentages),
                "action_verbs_used": len(set(found_verbs)),
                "action_verbs": list(set(found_verbs))[:10],
                "has_github": has_github,
                "has_linkedin": has_linkedin,
                "has_portfolio": has_portfolio,
                "professional_links_count": sum([has_github, has_linkedin, has_portfolio])
            }
        except Exception:
            return {
                "quantified_achievements": 0,
                "percentages_used": 0,
                "action_verbs_used": 0,
                "action_verbs": [],
                "has_github": False,
                "has_linkedin": False,
                "has_portfolio": False,
                "professional_links_count": 0
            }
    
    def _check_formatting_issues(self, resume_text: str) -> List[str]:
        """Check for common formatting issues"""
        issues = []
        
        try:
            # Check for excessive blank lines
            if re.search(r'\n\n\n+', resume_text):
                issues.append("Excessive blank lines found - may affect ATS parsing")
            
            # Check for email format
            if not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text):
                issues.append("Valid email address not found")
            
            # Check for phone number format
            if not re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text):
                issues.append("Phone number not found or improperly formatted")
            
            # Check for very long lines (potential formatting issue)
            lines = resume_text.split('\n')
            long_lines = [line for line in lines if len(line) > 120]
            if len(long_lines) > 3:
                issues.append("Some lines are too long - may cause display issues")
            
            # Check for inconsistent date formats
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b|\b[A-Za-z]+ \d{4}\b', resume_text)
            unique_date_lengths = set([len(d.replace('/', '').replace('-', '').replace(' ', '')) for d in dates])
            if len(unique_date_lengths) > 2:
                issues.append("Inconsistent date formats detected")
            
            # Check for missing periods in sentences
            sentences = re.split(r'[.!?]+', resume_text)
            sentences_without_periods = [s for s in sentences if len(s.strip()) > 50 and not s.strip().endswith(('.', '!', '?'))]
            if len(sentences_without_periods) > len(sentences) * 0.3:
                issues.append("Many sentences missing proper punctuation")
            
        except Exception:
            issues.append("Could not complete formatting analysis")
        
        return issues[:5]  # Limit to top 5 issues
    
    def _identify_strengths(self, resume_text: str, job_requirements: str) -> List[str]:
        """Identify resume strengths"""
        strengths = []
        resume_lower = resume_text.lower()
        
        try:
            # Technical skills presence
            tech_skills = ['python', 'sql', 'machine learning', 'git', 'data', 'analytics']
            tech_count = sum(1 for skill in tech_skills if skill in resume_lower)
            if tech_count >= 4:
                strengths.append("Strong technical skills foundation demonstrated")
            
            # Project presence
            if any(word in resume_lower for word in ['project', 'portfolio', 'github']):
                strengths.append("Includes relevant projects and work samples")
            
            # Quantified achievements
            numbers = re.findall(r'\b\d+\.?\d*%?\b', resume_text)
            if len(numbers) >= 5:
                strengths.append("Uses quantified achievements and metrics")
            
            # Education section
            education_terms = ['degree', 'university', 'college', 'bachelor', 'master', 'phd', 'certification']
            if any(word in resume_lower for word in education_terms):
                strengths.append("Clear educational background provided")
            
            # Contact information completeness
            has_email = '@' in resume_text
            has_phone = bool(re.search(r'\d{10}|\(\d{3}\)', resume_text))
            if has_email and has_phone:
                strengths.append("Complete contact information provided")
            
            # Professional online presence
            online_presence = sum([
                'github' in resume_lower,
                'linkedin' in resume_lower,
                'portfolio' in resume_lower
            ])
            if online_presence >= 2:
                strengths.append("Strong professional online presence")
            
            # Action-oriented language
            action_verbs = ['developed', 'implemented', 'created', 'designed', 'led', 'managed']
            action_count = sum(1 for verb in action_verbs if verb in resume_lower)
            if action_count >= 5:
                strengths.append("Uses action-oriented, impactful language")
            
            # Appropriate length
            word_count = len(resume_text.split())
            if 300 <= word_count <= 800:
                strengths.append("Appropriate resume length for ATS systems")
                
        except Exception:
            strengths.append("Resume structure appears professional")
        
        return strengths[:7]
    
    def _identify_weaknesses(self, resume_text: str, job_requirements: str) -> List[str]:
        """Identify resume weaknesses and areas for improvement"""
        weaknesses = []
        resume_lower = resume_text.lower()
        
        try:
            # Missing essential sections
            if not any(word in resume_lower for word in ['summary', 'objective', 'profile']):
                weaknesses.append("Missing professional summary or objective section")
            
            if not any(word in resume_lower for word in ['experience', 'work', 'employment']):
                weaknesses.append("Work experience section unclear or missing")
            
            if not any(word in resume_lower for word in ['skills', 'technical', 'competencies']):
                weaknesses.append("Technical skills section not clearly defined")
            
            # Keyword analysis
            keyword_analysis = self._analyze_keywords(resume_text, job_requirements)
            if keyword_analysis['keyword_density'] < 30:
                weaknesses.append(f"Low keyword match ({keyword_analysis['keyword_density']}%) for target role")
            
            # Content quality issues
            word_count = len(resume_text.split())
            if word_count < 250:
                weaknesses.append("Resume content too brief - needs more detail")
            elif word_count > 1000:
                weaknesses.append("Resume may be too lengthy - consider condensing")
            
            # Missing quantification
            numbers = re.findall(r'\b\d+\.?\d*%?\b', resume_text)
            if len(numbers) < 3:
                weaknesses.append("Lacks quantified achievements and measurable results")
            
            # Missing online presence
            if not any(word in resume_lower for word in ['github', 'linkedin', 'portfolio']):
                weaknesses.append("Missing professional online presence (GitHub, LinkedIn)")
            
            # Formatting issues
            formatting_issues = self._check_formatting_issues(resume_text)
            for issue in formatting_issues[:2]:  # Include top 2 formatting issues
                weaknesses.append(f"Formatting: {issue}")
            
            # Missing education
            if not any(word in resume_lower for word in ['education', 'degree', 'university', 'college']):
                weaknesses.append("Education section missing or unclear")
                
        except Exception:
            weaknesses.append("Unable to complete full weakness analysis")
        
        return weaknesses[:7]
    
    def _format_analysis_report(self, analysis: Dict) -> str:
        """Format the analysis results into a readable report"""
        try:
            # Safely get values with defaults
            ats_score = analysis.get('ats_score', 'N/A')
            keyword_analysis = analysis.get('keyword_analysis', {})
            structure_analysis = analysis.get('structure_analysis', {})
            content_analysis = analysis.get('content_analysis', {})
            strengths = analysis.get('strengths', [])
            weaknesses = analysis.get('weaknesses', [])
            formatting_issues = analysis.get('formatting_issues', [])
            
            report = f"""
🎯 COMPREHENSIVE RESUME ANALYSIS REPORT
{'='*50}

📊 ATS COMPATIBILITY SCORE: {ats_score}/10
{'🟢 Excellent' if ats_score >= 8 else '🟡 Good' if ats_score >= 6 else '🔴 Needs Improvement'}

🔍 KEYWORD ANALYSIS:
• Keyword Match Rate: {keyword_analysis.get('keyword_density', 0)}%
• Keywords Found ({len(keyword_analysis.get('found_keywords', []))}): {', '.join(keyword_analysis.get('found_keywords', [])[:8])}
• Top Missing Keywords: {', '.join(keyword_analysis.get('missing_keywords', [])[:8])}

📋 STRUCTURE ANALYSIS:
• Resume Sections: {', '.join(structure_analysis.get('sections_found', []))}
• Word Count: {structure_analysis.get('word_count', 0)} words
• Essential Sections: {'✅' if structure_analysis.get('has_experience') else '❌'} Experience | {'✅' if structure_analysis.get('has_skills') else '❌'} Skills | {'✅' if structure_analysis.get('has_education') else '❌'} Education

📈 CONTENT QUALITY:
• Quantified Achievements: {content_analysis.get('quantified_achievements', 0)} metrics found
• Action Verbs Used: {content_analysis.get('action_verbs_used', 0)} different verbs
• Professional Links: {'✅' if content_analysis.get('has_github') else '❌'} GitHub | {'✅' if content_analysis.get('has_linkedin') else '❌'} LinkedIn | {'✅' if content_analysis.get('has_portfolio') else '❌'} Portfolio

✅ STRENGTHS IDENTIFIED:
{chr(10).join([f"• {strength}" for strength in strengths]) if strengths else "• No specific strengths identified"}

⚠️ AREAS FOR IMPROVEMENT:
{chr(10).join([f"• {weakness}" for weakness in weaknesses]) if weaknesses else "• No major weaknesses identified"}

🔧 FORMATTING ISSUES:
{chr(10).join([f"• {issue}" for issue in formatting_issues]) if formatting_issues else "• No major formatting issues detected"}

💡 RECOMMENDATIONS:
• Focus on adding missing keywords naturally throughout your resume
• Include more quantified achievements with specific numbers and percentages
• Ensure all essential sections are clearly labeled and well-organized
• Add professional links (GitHub, LinkedIn) to showcase your work
• Keep resume length between 1-2 pages for optimal ATS compatibility

{'='*50}
Report generated by AI Resume Analyzer
"""
            return report
            
        except Exception as e:
            return f"""
ERROR IN ANALYSIS REPORT
{'='*30}
An error occurred while formatting the analysis report: {str(e)}

Raw Analysis Data:
{str(analysis)}
"""
