#!/usr/bin/env python3
"""
Resume Analyzer & Enhancer - Application Launcher
Run this file to start the application
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'crewai', 'fastapi', 'uvicorn', 'python-docx', 
        'PyPDF2', 'reportlab', 'google-generativeai', 'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        # Install missing packages
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'outputs', 
        'config',
        'src/ui/static',
        'src/ui/templates',
        'src/tools',
        'src/knowledge'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_config_files():
    """Create configuration files if they don't exist"""
    
    # Create agents.yaml if it doesn't exist
    agents_config = """resume_analyzer:
  role: "Senior Resume Analyst"
  goal: "Analyze resumes against job requirements and identify gaps, strengths, and ATS compatibility issues"
  backstory: "You are an experienced HR professional and ATS expert with deep knowledge of AI/Data Science hiring requirements."
  verbose: true
  allow_delegation: false
  max_iter: 15
  memory: true

resume_enhancer:
  role: "Resume Enhancement Specialist" 
  goal: "Provide specific, actionable improvement suggestions for resumes"
  backstory: "You are a career coach specializing in AI/Data Science roles with expertise in ATS optimization."
  verbose: true
  allow_delegation: false
  max_iter: 15
  memory: true

resume_generator:
  role: "Professional Resume Writer"
  goal: "Generate polished, ATS-friendly resumes based on improvement suggestions"
  backstory: "You are a professional resume writer with expertise in creating clean, modern, ATS-optimized resumes."
  verbose: true
  allow_delegation: false
  max_iter: 20
  memory: true

web_researcher:
  role: "Job Market Research Specialist"
  goal: "Search and analyze current job postings for AI/Data Science roles"
  backstory: "You are a market research expert specializing in tech job trends and requirements."
  verbose: true
  allow_delegation: false
  max_iter: 10
  memory: true"""

    # Create tasks.yaml if it doesn't exist
    tasks_config = """analyze_resume:
  description: >
    Analyze the uploaded resume against the provided requirements focusing on technical skills alignment,
    ATS compatibility, keyword optimization, project relevance, and structure.
  expected_output: >
    A comprehensive analysis report including strengths, weaknesses, ATS compatibility score,
    missing keywords list, and formatting recommendations.
  agent: resume_analyzer

enhance_resume:
  description: >
    Based on analysis results, provide specific improvement suggestions including keyword additions,
    project description improvements, skills section enhancements, and ATS optimization techniques.
  expected_output: >
    Detailed improvement suggestions with specific text replacements, keyword additions,
    project rewrites, skills prioritization, and achievement quantification examples.
  agent: resume_enhancer
  context:
    - analyze_resume

generate_resume:
  description: >
    Generate an enhanced version of the original resume incorporating improvement suggestions
    while maintaining authenticity and ensuring ATS compatibility.
  expected_output: >
    A complete enhanced resume with improved content, ATS-optimized formatting,
    and professional presentation suitable for DOCX/PDF generation.
  agent: resume_generator
  context:
    - analyze_resume
    - enhance_resume

research_jobs:
  description: >
    Search for current AI/Data Science entry-level job postings to identify most demanded skills,
    common requirements, popular technologies, and certification preferences.
  expected_output: >
    A comprehensive job market analysis including top skills, requirements, emerging technologies,
    preferred certifications, and typical project types mentioned in job postings.
  agent: web_researcher"""

    # Write config files
    config_files = {
        'config/agents.yaml': agents_config,
        'config/tasks.yaml': tasks_config
    }
    
    for file_path, content in config_files.items():
        if not Path(file_path).exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"📝 Created config file: {file_path}")

def main():
    """Main function to setup and run the application"""
    print("🚀 Resume Analyzer & Enhancer - Setup and Launch")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages manually.")
        return
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Create config files
    print("\n⚙️ Setting up configuration...")
    create_config_files()
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/tools/__init__.py'
    ]
    
    for init_file in init_files:
        if not Path(init_file).exists():
            Path(init_file).touch()
    
    print("\n✅ Setup completed successfully!")
    print("\n🌐 Starting application server...")
    print("📝 Access the application at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Use uvicorn.run with string reference for proper reload functionality
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()