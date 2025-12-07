from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from ai_model import predict_and_explain
from auth import authenticate
from models import Session, Log, User
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import shutil

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Login endpoint (shared)
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    user = authenticate(username, password, role)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return {"message": "Logged in", "user_id": user.id, "role": role}  # Use sessions/JWT in prod

# Admin Module
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/upload_model")
async def upload_model(file: UploadFile = File(...)):
    with open("best_model.pth", "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "Model uploaded"}

# Doctor Module
@app.get("/doctor", response_class=HTMLResponse)
async def doctor_page(request: Request):
    return templates.TemplateResponse("doctor.html", {"request": request})

@app.post("/doctor/diagnose")
async def diagnose(image: UploadFile = File(...), user_id: int = Form(...)):
    image_path = f"uploads/{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    pred, conf, heatmap = predict_and_explain(image_path)
    
    # Log
    session = Session()
    log = Log(user_id=user_id, action=f"Diagnosis: {pred}")
    session.add(log)
    session.commit()
    session.close()
    
    # Generate report
    report_path = f"reports/report_{user_id}.pdf"
    c = canvas.Canvas(report_path, pagesize=letter)
    c.drawString(100, 750, f"Diagnosis: {pred} (Confidence: {conf:.2%})")
    c.drawImage(image_path, 100, 600, width=200, height=200)
    c.drawImage(heatmap, 100, 300, width=200, height=200)
    c.save()
    
    return {"prediction": pred, "confidence": conf, "heatmap": heatmap, "report": report_path}

@app.get("/download_report/{path}")
async def download_report(path: str):
    return FileResponse(f"reports/{path}")

# User Module (similar to Doctor, but limited)
@app.get("/user", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("user.html", {"request": request})

# AI Diagnosis (called internally)

# Run: uvicorn app.py:app --reload