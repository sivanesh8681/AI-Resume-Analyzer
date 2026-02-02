# main.py - Final Corrected Version
import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your custom modules
from src.crew import ResumeAnalyzerCrew

app = FastAPI(title="AI Resume Analyzer & Enhancer")

# 1. Setup Directories
directories = ["uploads", "outputs", "src/ui/static", "src/ui/templates"]
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

# 2. Middleware & Static Files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
templates = Jinja2Templates(directory="src/ui/templates")

# 3. Global Storage (Use Redis/DB for production)
analysis_storage = {}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    mode: str = Form(...)
):
    """
    Main entry point for Analysis. 
    Fixes the 'Analysis Failed' error by normalizing mode strings.
    """
    try:
        print(f"🚀 Processing: {file.filename} | Mode: {mode}")
        
        # Save uploaded file safely
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize the AI Crew
        crew = ResumeAnalyzerCrew()
        
        # --- THE FIX: Robust Mode Matching ---
        # Frontends often send "Quick Analysis" or "web_enhanced"
        mode_val = mode.lower()
        
        if any(x in mode_val for x in ["static", "quick"]):
            print("🔍 Executing: Quick Static Analysis")
            result = await crew.analyze_with_static_knowledge(file_path)
        elif "web" in mode_val:
            print("🌐 Executing: Web-Enhanced Analysis")
            result = await crew.analyze_with_web_search(file_path)
        else:
            print("⚠️ Mode mismatch, defaulting to static")
            result = await crew.analyze_with_static_knowledge(file_path)
        
        # Store analysis for the 'Generate' feature
        analysis_storage[file.filename] = {
            'analysis': result,
            'file_path': file_path
        }
        
        return JSONResponse(content={
            "status": "success",
            "analysis": result, # Contains ats_score, strengths, missing_keywords
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server side failure: {str(e)}"}
        )

@app.post("/generate-resume")
async def generate_resume(request: Request):
    """
    Triggers the DOCX generation logic in src/crew.py
    """
    try:
        data = await request.json()
        filename = data.get("filename")
        
        if filename not in analysis_storage:
            raise HTTPException(status_code=404, detail="Analysis session expired")
            
        stored = analysis_storage[filename]
        crew = ResumeAnalyzerCrew()
        
        # Call the generation method in your crew.py
        output_filename = await crew.generate_enhanced_resume(stored['file_path'], stored['analysis'])
        
        if output_filename:
            return {"status": "success", "download_url": f"/download/{output_filename}"}
        raise Exception("File creation failed")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            filename=filename
        )
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
async def health():
    return {"status": "online", "api": "Gemini 1.5 Flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)