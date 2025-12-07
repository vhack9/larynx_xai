**(
first run and install python packages in the requirements.txt to install python packages 

or run

For laptops (CPU only, minimal size):
pip install fastapi uvicorn[standard] torch torchvision --index-url https://download.pytorch.org/whl/cpu easyfsl grad-cam Pillow sqlalchemy Jinja2 python-multipart reportlab numpy passlib[bcrypt] python-jose[cryptography]

if have NVIDIA GPU and want faster training/inference:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

)**

Run python train_model.py to get the model.

Add sample users to DB (manual insert via SQLite browser: admin/admin, doctor/doctor, user/user).

Run uvicorn app.py:app --reload.

Access http://127.0.0.1:8000/login, log in by role.

Test: Upload image as doctor → Get prediction, heatmap, PDF report.
