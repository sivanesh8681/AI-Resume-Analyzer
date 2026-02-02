# test_resume_generation.py - Test script to verify resume generation

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.tools.file_tools import ResumeGenerator, EnhancedResumeContentGenerator

def test_resume_generation():
    """Test the resume generation functionality"""
    print("🧪 Testing Resume Generation")
    print("=" * 40)
    
    # Create test directories
    os.makedirs("outputs", exist_ok=True)
    
    # Sample resume content
    sample_content = """
JOHN DOE
Email: john.doe@email.com | Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

PROFESSIONAL SUMMARY
Data Science enthusiast with experience in Python and SQL.

TECHNICAL SKILLS
• Python
• SQL
• Excel

PROJECTS
Data Analysis Project
• Analyzed sales data using Python
• Created visualizations

EDUCATION
Bachelor of Science in Computer Science
University Name, 2024
"""
    
    # Test basic generation
    print("1. Testing basic resume generation...")
    try:
        generator = ResumeGenerator()
        output_path = "outputs/test_resume.docx"
        
        result = generator.generate_docx(sample_content, output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Resume generated successfully")
            print(f"   File: {output_path}")
            print(f"   Size: {file_size} bytes")
        else:
            print(f"❌ Resume generation failed - file not created")
            
    except Exception as e:
        print(f"❌ Basic generation error: {e}")
    
    # Test enhanced content generation
    print("\n2. Testing enhanced content generation...")
    try:
        analysis_data = {
            'missing_keywords': ['Machine Learning', 'Pandas', 'NumPy', 'Git', 'TensorFlow'],
            'strengths': ['Good technical foundation'],
            'weaknesses': ['Missing quantified achievements', 'Limited project descriptions']
        }
        
        improvements = ['Add more technical keywords', 'Quantify achievements']
        
        enhanced_content = EnhancedResumeContentGenerator.enhance_resume_content(
            sample_content, analysis_data, improvements
        )
        
        if enhanced_content and len(enhanced_content) > len(sample_content):
            print("✅ Content enhancement successful")
            print(f"   Original length: {len(sample_content)} chars")
            print(f"   Enhanced length: {len(enhanced_content)} chars")
            
            # Generate enhanced resume
            enhanced_output = "outputs/test_enhanced_resume.docx" 
            generator.generate_docx(enhanced_content, enhanced_output)
            
            if os.path.exists(enhanced_output):
                enhanced_size = os.path.getsize(enhanced_output)
                print(f"✅ Enhanced resume generated")
                print(f"   File: {enhanced_output}")
                print(f"   Size: {enhanced_size} bytes")
            else:
                print("❌ Enhanced resume file not created")
                
        else:
            print("❌ Content enhancement failed")
            
    except Exception as e:
        print(f"❌ Enhanced generation error: {e}")
    
    # Test with minimal content
    print("\n3. Testing with minimal content...")
    try:
        minimal_content = "John Doe\nSoftware Developer"
        minimal_output = "outputs/test_minimal_resume.docx"
        
        generator.generate_docx(minimal_content, minimal_output)
        
        if os.path.exists(minimal_output):
            minimal_size = os.path.getsize(minimal_output)
            print(f"✅ Minimal resume generated")
            print(f"   File: {minimal_output}")
            print(f"   Size: {minimal_size} bytes")
        else:
            print("❌ Minimal resume generation failed")
            
    except Exception as e:
        print(f"❌ Minimal generation error: {e}")
    
    print("\n🎯 Test completed!")
    print(f"📁 Check the 'outputs' directory for generated files")

def test_content_parsing():
    """Test content parsing functionality"""
    print("\n🔍 Testing Content Parsing")
    print("=" * 30)
    
    sample_content = """
JANE SMITH
jane.smith@email.com | (555) 987-6543

SUMMARY
Experienced data analyst with Python skills.

SKILLS
• Python
• SQL
• Tableau

EXPERIENCE
Data Analyst at Company XYZ
• Analyzed customer data
• Created reports

PROJECTS
Sales Analysis
• Used Python for analysis
• Generated insights
"""
    
    try:
        generator = ResumeGenerator()
        sections = generator._parse_resume_content(sample_content)
        
        print("✅ Content parsed successfully:")
        for section, content in sections.items():
            print(f"   {section}: {len(content)} chars")
        
        if len(sections) >= 3:
            print("✅ Parsing test passed")
        else:
            print("⚠️ Parsing may have issues")
            
    except Exception as e:
        print(f"❌ Parsing error: {e}")

if __name__ == "__main__":
    test_resume_generation()
    test_content_parsing()