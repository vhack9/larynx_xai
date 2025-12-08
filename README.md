**(
#,Package Name  ,  Pip Install Command,Approx. Size (MB),Purpose in LarynxXAI Project,  Essential?

1,fastapi ,  pip install fastapi,  ~3 MB,"Modern, fast web framework (replaces Flask/Django) — handles all API routes (login, upload, diagnosis)" ,  Yes

2,uvicorn  ,  pip install uvicorn[standard],  ~15 MB,ASGI server to run FastAPI (production-ready), Yes

3,torch + torchvision  , pip install torch torchvision,  "80–2,500 MB",Core PyTorch library + image utilities. Size depends on CPU vs CUDA (GPU).  Use CPU version for laptops.,Yes

,→ CPU-only (recommended)  ,  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu,~120 MB,,
,→ GPU (CUDA 12.1)  ,  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121,~2.5 GB,,

4,easyfsl  ,  pip install easyfsl,  ~2 MB,"Simplest library for few-shot learning (Prototypical Networks, Matching Networks, etc.) in <10 lines" ,  Yes

5,pytorch-grad-cam  ,  pip install grad-cam,  ~5 MB,Generates beautiful Grad-CAM heatmaps for explainability — the core XAI feature ,   Yes

6,Pillow ,  pip install Pillow,  ~25 MB,"Image loading, saving, and basic processing (open 100×100 patches)"  ,  Yes

7,SQLAlchemy ,  pip install sqlalchemy,  ~12 MB,"ORM for SQLite/PostgreSQL — manages users, logs, roles", Yes

8,Jinja2  ,  pip install Jinja2,  ~10 MB,HTML templating engine (used with FastAPI to render login/admin/doctor pages), Yes

9,python-multipart ,  pip install python-multipart ,  ~1 MB,Required for file uploads in FastAPI (image upload forms)  , Yes

10,reportlab ,  pip install reportlab,  ~40 MB,Generates downloadable PDF diagnostic reports with image + heatmap + result ,  Yes

11,opencv-python (optional) ,  pip install opencv-python ,  ~80 MB,Used only to save heatmap overlay as PNG (cv2.imwrite). Pillow can replace it if you want smaller size. ,  Optional

12,numpy ,  pip install numpy ,  ~25–40 MB,"Numerical operations everywhere (image arrays, model inputs)" ,  Yes

13,bcrypt or passlib ,  pip install passlib[bcrypt],  ~5 MB,Secure password hashing (highly recommended for production login), Recommended

14,python-jose[cryptography] ,  pip install python-jose[cryptography],  ~15 MB,JWT token generation for proper session/auth (instead of plain text passwords) ,  Recommended

15,starlette ,  (already pulled by FastAPI) , — ,  Underlying framework for FastAPI — no need to install separately,—


)**

Run python train_model.py to get the model.

Add sample users to DB (manual insert via SQLite browser: admin/admin, doctor/doctor, user/user).

Run uvicorn app.py:app --reload.

Access http://127.0.0.1:8000/login, log in by role.

Test: Upload image as doctor → Get prediction, heatmap, PDF report.
