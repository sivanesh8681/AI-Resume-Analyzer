import requests
import re
from typing import Any, Optional, Type, List, Dict
from crewai.tools import BaseTool


from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import json

class JobSearchInput(BaseModel):
    query: str = Field(default="AI Data Science entry level jobs", description="Job search query")
    num_results: int = Field(default=8, description="Number of job results to fetch")

class JobSearchTool(BaseTool):
    name: str = "Job Search Tool"
    description: str = "Search for current job postings and extract requirements for AI/Data Science roles"
    args_schema: Type[BaseModel] = JobSearchInput

    def _run(self, query: str = "AI Data Science entry level jobs", num_results: int = 8) -> str:
        """Search for job postings and extract requirements"""
        try:
            # Use multiple search engines/job boards
            all_jobs = []
            
            # Search using different sources
            indeed_jobs = self._search_indeed(query, num_results//2)
            linkedin_jobs = self._search_linkedin(query, num_results//2)
            
            all_jobs.extend(indeed_jobs)
            all_jobs.extend(linkedin_jobs)
            
            # Analyze and summarize findings
            analysis = self._analyze_job_requirements(all_jobs)
            
            return self._format_job_analysis(analysis, all_jobs[:6])
            
        except Exception as e:
            return f"Error searching jobs: {str(e)}"
    
    def _search_indeed(self, query: str, num_results: int) -> List[Dict]:
        """Search Indeed for job postings (simplified simulation)"""
        # Note: In production, you'd use Indeed's API or proper web scraping
        # For this PoC, we'll simulate realistic job data
        
        simulated_jobs = [
            {
                "title": "Junior Data Scientist",
                "company": "TechCorp Inc", 
                "requirements": "Python, SQL, Machine Learning, Pandas, Scikit-learn, Statistics, Bachelor's degree, 0-2 years experience",
                "skills": ["Python", "SQL", "Machine Learning", "Pandas", "Scikit-learn", "Statistics", "Git", "Jupyter"]
            },
            {
                "title": "Entry Level AI Engineer",
                "company": "DataSolutions Ltd",
                "requirements": "Python, TensorFlow, PyTorch, Deep Learning, Computer Vision, NLP, Master's preferred, Portfolio projects",
                "skills": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "Computer Vision", "NLP", "Docker", "AWS"]
            },
            {
                "title": "Data Analyst - Fresher",
                "company": "Analytics Pro",
                "requirements": "SQL, Excel, Tableau, Power BI, Statistics, Data Visualization, Communication skills, Bachelor's degree",
                "skills": ["SQL", "Excel", "Tableau", "Power BI", "Statistics", "Data Visualization", "Python", "R"]
            },
            {
                "title": "Machine Learning Intern",
                "company": "AI Innovations",
                "requirements": "Python, Scikit-learn, Pandas, NumPy, Matplotlib, Linear Algebra, Calculus, GitHub portfolio",
                "skills": ["Python", "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Git", "Jupyter", "Statistics"]
            }
        ]
        
        return simulated_jobs[:num_results]
    
    def _search_linkedin(self, query: str, num_results: int) -> List[Dict]:
        """Search LinkedIn for job postings (simplified simulation)"""
        simulated_jobs = [
            {
                "title": "Associate Data Scientist",
                "company": "BigData Corp",
                "requirements": "Python, R, SQL, Hadoop, Spark, Machine Learning, Cloud platforms (AWS/Azure), Agile methodology",
                "skills": ["Python", "R", "SQL", "Hadoop", "Spark", "Machine Learning", "AWS", "Azure", "Agile"]
            },
            {
                "title": "AI Developer - Entry Level",
                "company": "Neural Networks Inc",
                "requirements": "Python, TensorFlow, Keras, OpenCV, NLTK, REST APIs, Docker, Kubernetes, CS degree",
                "skills": ["Python", "TensorFlow", "Keras", "OpenCV", "NLTK", "REST API", "Docker", "Kubernetes"]
            },
            {
                "title": "Junior Business Intelligence Analyst",
                "company": "DataViz Solutions",
                "requirements": "SQL, Tableau, Power BI, Excel, ETL processes, Data modeling, Statistical analysis, Dashboard creation",
                "skills": ["SQL", "Tableau", "Power BI", "Excel", "ETL", "Data Modeling", "Statistics", "Dashboard"]
            },
            {
                "title": "Graduate Data Engineer",
                "company": "CloudTech Systems",
                "requirements": "Python, SQL, Apache Airflow, Apache Kafka, MongoDB, PostgreSQL, Git, CI/CD, Linux",
                "skills": ["Python", "SQL", "Airflow", "Kafka", "MongoDB", "PostgreSQL", "Git", "CI/CD", "Linux"]
            }
        ]
        
        return simulated_jobs[:num_results]
    
    def _analyze_job_requirements(self, jobs: List[Dict]) -> Dict:
        """Analyze job requirements to find common patterns"""
        all_skills = []
        all_requirements = []
        
        for job in jobs:
            all_skills.extend(job.get('skills', []))
            all_requirements.append(job.get('requirements', ''))
        
        # Count skill frequency
        skill_count = {}
        for skill in all_skills:
            skill_count[skill] = skill_count.get(skill, 0) + 1
        
        # Sort by frequency
        top_skills = sorted(skill_count.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Analyze requirements text
        combined_requirements = ' '.join(all_requirements).lower()
        
        # Extract common patterns
        degree_requirements = self._extract_degree_requirements(combined_requirements)
        experience_requirements = self._extract_experience_requirements(combined_requirements)
        certifications = self._extract_certifications(combined_requirements)
        
        return {
            "top_skills": top_skills,
            "degree_requirements": degree_requirements,
            "experience_requirements": experience_requirements,
            "certifications": certifications,
            "total_jobs_analyzed": len(jobs)
        }
    
    def _extract_degree_requirements(self, text: str) -> List[str]:
        """Extract degree requirements from job text"""
        degree_patterns = [
            r'bachelor[\'s]*\s+degree', r'master[\'s]*\s+degree', r'phd', r'doctorate',
            r'computer science', r'data science', r'statistics', r'mathematics',
            r'engineering', r'physics', r'economics'
        ]
        
        found_degrees = []
        for pattern in degree_patterns:
            if re.search(pattern, text):
                found_degrees.append(pattern.replace(r'[\'s]*\s+', ' ').replace(r'\s+', ' '))
        
        return list(set(found_degrees))[:5]
    
    def _extract_experience_requirements(self, text: str) -> List[str]:
        """Extract experience requirements"""
        exp_patterns = [
            r'0-2 years', r'entry level', r'junior', r'fresher', r'graduate',
            r'internship', r'no experience', r'recent graduate'
        ]
        
        found_exp = []
        for pattern in exp_patterns:
            if re.search(pattern, text):
                found_exp.append(pattern)
        
        return list(set(found_exp))
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certification requirements"""
        cert_patterns = [
            r'aws certified', r'azure certified', r'google cloud', r'tensorflow certified',
            r'tableau certified', r'microsoft certified', r'oracle certified',
            r'coursera', r'udacity', r'edx'
        ]
        
        found_certs = []
        for pattern in cert_patterns:
            if re.search(pattern, text):
                found_certs.append(pattern.replace(r'\s+', ' '))
        
        return list(set(found_certs))[:5]
    
    def _format_job_analysis(self, analysis: Dict, sample_jobs: List[Dict]) -> str:
        """Format job analysis into readable report"""
        
        top_skills_str = '\n'.join([f"- {skill}: {count} jobs" for skill, count in analysis['top_skills'][:15]])
        
        sample_jobs_str = '\n'.join([
            f"- {job['title']} at {job['company']}\n  Requirements: {job['requirements'][:100]}..."
            for job in sample_jobs[:4]
        ])
        
        report = f"""
CURRENT JOB MARKET ANALYSIS
===========================

Total Jobs Analyzed: {analysis['total_jobs_analyzed']}

TOP IN-DEMAND SKILLS:
{top_skills_str}

DEGREE REQUIREMENTS:
{', '.join(analysis['degree_requirements']) if analysis['degree_requirements'] else 'Varies by position'}

EXPERIENCE REQUIREMENTS:
{', '.join(analysis['experience_requirements']) if analysis['experience_requirements'] else 'Entry level positions available'}

PREFERRED CERTIFICATIONS:
{', '.join(analysis['certifications']) if analysis['certifications'] else 'Not commonly required for entry-level'}

SAMPLE JOB POSTINGS:
{sample_jobs_str}

KEY INSIGHTS:
- Python is the most demanded skill across AI/DS roles
- SQL knowledge is essential for most data-related positions
- Machine Learning experience highly valued
- Portfolio projects often preferred over extensive work experience
- Cloud platform knowledge (AWS/Azure) increasingly important
- Statistics and mathematics foundation crucial
"""
        
        return report


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Generic web search tool for finding information"
    args_schema: Type[BaseModel] = JobSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """Perform web search using a simple search simulation"""
        try:
            # For PoC, we'll simulate search results
            # In production, integrate with actual search APIs (Google, Bing, etc.)
            
            if 'job' in query.lower() or 'hiring' in query.lower():
                # Delegate to job search tool
                job_tool = JobSearchTool()
                return job_tool._run(query, num_results)
            
            # Generic search simulation
            simulated_results = [
                {
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example{i+1}.com",
                    "snippet": f"This is a simulated search result for {query}. Contains relevant information about the topic."
                }
                for i in range(num_results)
            ]
            
            formatted_results = '\n'.join([
                f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n"
                for result in simulated_results
            ])
            
            return f"Search Results for '{query}':\n\n{formatted_results}"
            
        except Exception as e:
            return f"Search error: {str(e)}"