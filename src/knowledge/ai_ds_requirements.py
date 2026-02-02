"""
Static Knowledge Base for AI/Data Science Job Requirements
This contains curated information about common requirements for fresher and entry-level positions
"""

AI_DS_REQUIREMENTS = {
    "core_technical_skills": {
        "programming_languages": {
            "essential": ["Python", "SQL"],
            "preferred": ["R", "Java", "Scala", "JavaScript"],
            "emerging": ["Julia", "Go"]
        },
        "data_manipulation": {
            "essential": ["Pandas", "NumPy", "Excel"],
            "preferred": ["Dask", "Polars", "Apache Spark"]
        },
        "machine_learning": {
            "essential": ["Scikit-learn", "Statistics", "Linear Algebra"],
            "preferred": ["TensorFlow", "PyTorch", "Keras", "XGBoost"],
            "specialized": ["Hugging Face", "OpenCV", "NLTK", "spaCy"]
        },
        "data_visualization": {
            "essential": ["Matplotlib", "Seaborn", "Tableau"],
            "preferred": ["Power BI", "Plotly", "D3.js", "Bokeh"]
        },
        "databases": {
            "essential": ["MySQL", "PostgreSQL"],
            "preferred": ["MongoDB", "Redis", "Cassandra", "Neo4j"]
        },
        "cloud_platforms": {
            "essential": ["AWS basics", "Google Cloud basics"],
            "preferred": ["Azure", "Docker", "Kubernetes"],
            "tools": ["S3", "EC2", "BigQuery", "Cloud ML"]
        }
    },
    
    "mathematical_foundations": {
        "essential": [
            "Statistics and Probability",
            "Linear Algebra", 
            "Calculus (basic)",
            "Descriptive Statistics",
            "Hypothesis Testing",
            "Regression Analysis"
        ],
        "preferred": [
            "Multivariate Calculus",
            "Optimization Theory",
            "Bayesian Statistics",
            "Time Series Analysis"
        ]
    },
    
    "domain_knowledge": {
        "machine_learning_concepts": [
            "Supervised Learning",
            "Unsupervised Learning", 
            "Classification",
            "Regression",
            "Clustering",
            "Feature Engineering",
            "Model Evaluation",
            "Cross-validation",
            "Overfitting/Underfitting",
            "Bias-Variance Tradeoff"
        ],
        "deep_learning_concepts": [
            "Neural Networks",
            "CNN (Computer Vision)",
            "RNN/LSTM (NLP)",
            "Transfer Learning",
            "Gradient Descent",
            "Backpropagation"
        ],
        "data_science_lifecycle": [
            "Data Collection",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Feature Selection",
            "Model Building",
            "Model Deployment",
            "A/B Testing"
        ]
    },
    
    "tools_and_technologies": {
        "development_tools": {
            "essential": ["Git", "GitHub", "Jupyter Notebook", "IDE (PyCharm/VS Code)"],
            "preferred": ["Docker", "Apache Airflow", "MLflow", "DVC"]
        },
        "data_processing": {
            "essential": ["Pandas", "NumPy", "Excel"],
            "preferred": ["Apache Spark", "Hadoop", "Kafka", "ETL tools"]
        },
        "deployment": {
            "preferred": ["Flask", "FastAPI", "Streamlit", "Heroku", "AWS Lambda"]
        }
    },
    
    "soft_skills": {
        "essential": [
            "Problem-solving",
            "Critical thinking", 
            "Communication skills",
            "Team collaboration",
            "Attention to detail",
            "Curiosity and learning mindset"
        ],
        "preferred": [
            "Business acumen",
            "Project management",
            "Presentation skills",
            "Stakeholder management",
            "Domain expertise"
        ]
    },
    
    "education_requirements": {
        "minimum": [
            "Bachelor's degree in Computer Science, Statistics, Mathematics, Engineering, or related field",
            "Strong academic foundation in quantitative subjects"
        ],
        "preferred": [
            "Master's degree in Data Science, Machine Learning, or related field",
            "Relevant coursework in AI/ML, Statistics, Database Systems",
            "Online certifications from Coursera, edX, Udacity"
        ]
    },
    
    "experience_expectations": {
        "fresher_0_1_years": [
            "Strong portfolio of personal/academic projects",
            "Internship experience (preferred but not mandatory)",
            "Participation in hackathons, competitions (Kaggle, etc.)",
            "Open source contributions",
            "Research publications (for advanced roles)"
        ],
        "entry_level_1_2_years": [
            "1-2 years of relevant experience or internships",
            "Demonstrated project experience with real datasets",
            "Understanding of business applications of AI/ML",
            "Experience with end-to-end project delivery"
        ]
    },
    
    "project_portfolio_requirements": {
        "essential_projects": [
            "End-to-end ML project with real dataset",
            "Data analysis and visualization project",
            "Web scraping or API integration project",
            "Database design and querying project"
        ],
        "preferred_projects": [
            "Deep learning project (Computer Vision or NLP)",
            "Time series analysis project",
            "Recommendation system",
            "A/B testing analysis",
            "Dashboard or web application",
            "Deployed ML model (API or web app)"
        ],
        "project_documentation": [
            "Clear README files",
            "Well-commented code",
            "Data source attribution", 
            "Results interpretation",
            "Future improvements section"
        ]
    },
    
    "certifications": {
        "highly_valued": [
            "Google Cloud Professional Data Engineer",
            "AWS Certified Machine Learning - Specialty",
            "Microsoft Azure AI Engineer Associate",
            "Google Cloud Professional ML Engineer",
            "Tableau Desktop Certified Associate"
        ],
        "good_to_have": [
            "TensorFlow Developer Certificate",
            "IBM Data Science Professional Certificate",
            "Microsoft Power BI Data Analyst Associate",
            "SAS Certified Specialist",
            "Databricks Certified Associate Developer"
        ]
    },
    
    "ats_optimization_guidelines": {
        "keyword_density": "Include relevant technical keywords naturally throughout resume",
        "section_headers": ["Summary", "Skills", "Experience", "Projects", "Education"],
        "formatting": {
            "use_standard_fonts": ["Arial", "Calibri", "Times New Roman"],
            "avoid": ["Images", "Graphics", "Complex tables", "Special characters"],
            "structure": "Use clear headings, bullet points, consistent formatting"
        },
        "file_format": "PDF preferred, ensure text is selectable",
        "length": "1-2 pages maximum for entry-level positions"
    },
    
    "common_job_titles": {
        "entry_level": [
            "Junior Data Scientist",
            "Data Analyst",
            "Business Intelligence Analyst", 
            "Machine Learning Engineer (Junior)",
            "Data Engineer (Associate)",
            "Research Analyst",
            "Quantitative Analyst"
        ],
        "internships": [
            "Data Science Intern",
            "ML Engineering Intern",
            "Analytics Intern",
            "AI Research Intern",
            "Business Intelligence Intern"
        ]
    },
    
    "industry_applications": {
        "finance": ["Risk modeling", "Fraud detection", "Algorithmic trading", "Credit scoring"],
        "healthcare": ["Medical imaging", "Drug discovery", "Patient outcome prediction", "Genomics"],
        "technology": ["Recommendation systems", "Search optimization", "User behavior analysis", "Product analytics"],
        "retail": ["Customer segmentation", "Price optimization", "Inventory management", "Demand forecasting"],
        "manufacturing": ["Predictive maintenance", "Quality control", "Supply chain optimization", "Process optimization"]
    },
    
    "salary_expectations": {
        "fresher_india": {
            "data_analyst": "3-6 LPA",
            "data_scientist": "4-8 LPA", 
            "ml_engineer": "5-10 LPA",
            "ai_engineer": "6-12 LPA"
        },
        "entry_level_us": {
            "data_analyst": "$50K-70K",
            "data_scientist": "$70K-95K",
            "ml_engineer": "$80K-110K",
            "ai_engineer": "$85K-120K"
        }
    }
}

# Additional utility functions for knowledge base
def get_skill_priority(role_type="data_scientist"):
    """Get prioritized skills for specific role types"""
    
    role_priorities = {
        "data_scientist": {
            "must_have": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "Scikit-learn"],
            "nice_to_have": ["R", "TensorFlow", "Cloud platforms", "Big Data tools"]
        },
        "data_analyst": {
            "must_have": ["SQL", "Excel", "Statistics", "Data Visualization", "Python/R"],
            "nice_to_have": ["Tableau", "Power BI", "Database management", "ETL"]
        },
        "ml_engineer": {
            "must_have": ["Python", "Machine Learning", "TensorFlow/PyTorch", "Git", "Cloud platforms"],
            "nice_to_have": ["Docker", "Kubernetes", "MLOps", "Model deployment"]
        },
        "ai_engineer": {
            "must_have": ["Python", "Deep Learning", "TensorFlow", "Neural Networks", "Computer Vision/NLP"],
            "nice_to_have": ["Research experience", "Publications", "Advanced mathematics"]
        }
    }
    
    return role_priorities.get(role_type, role_priorities["data_scientist"])

def get_ats_keywords():
    """Get comprehensive list of ATS-friendly keywords"""
    all_skills = []
    
    # Extract all technical skills
    for category in AI_DS_REQUIREMENTS["core_technical_skills"].values():
        if isinstance(category, dict):
            for skill_level in category.values():
                all_skills.extend(skill_level)
        else:
            all_skills.extend(category)
    
    # Add domain knowledge terms
    all_skills.extend(AI_DS_REQUIREMENTS["domain_knowledge"]["machine_learning_concepts"])
    all_skills.extend(AI_DS_REQUIREMENTS["domain_knowledge"]["deep_learning_concepts"])
    
    # Add mathematical foundations
    all_skills.extend(AI_DS_REQUIREMENTS["mathematical_foundations"]["essential"])
    
    return list(set(all_skills))

def get_project_suggestions():
    """Get project suggestions for portfolio building"""
    return {
        "beginner_projects": [
            "Exploratory Data Analysis on public dataset",
            "Simple linear/logistic regression project",
            "Data visualization dashboard",
            "Web scraping and analysis project"
        ],
        "intermediate_projects": [
            "End-to-end ML pipeline with model deployment",
            "Time series forecasting project", 
            "Computer vision image classification",
            "Natural language processing sentiment analysis",
            "Recommendation system implementation"
        ],
        "advanced_projects": [
            "Deep learning project with custom architecture",
            "Multi-model ensemble for complex prediction",
            "Real-time data processing pipeline",
            "A/B testing framework implementation",
            "MLOps pipeline with monitoring and retraining"
        ]
    }