import subprocess
import os
import time
import webbrowser
import threading

# Paths
FRONTEND_DIR = "Frontend"
BACKEND_DIR = "Backend"
REACT_DEV_URL = "http://localhost:5173/"  # Adjust if your frontend uses a different port
FASTAPI_URL = "http://127.0.0.1:8000/docs"  # Optional: to open FastAPI docs

# --------------------- Frontend ---------------------

def start_frontend():
    print("🚀 Starting Frontend...")
    os.chdir(FRONTEND_DIR)
    
    # Install dependencies
    subprocess.run(["npm", "install"], check=True)

    # Start React dev server
    threading.Thread(target=lambda: subprocess.run(["npm", "run", "dev"], shell=True)).start()
    
    time.sleep(5)  # Wait for server to start
    webbrowser.open(REACT_DEV_URL)
    
    os.chdir("..")

# --------------------- Backend ----------------------

def start_backend():
    print("🚀 Starting Backend...")
    os.chdir(BACKEND_DIR)

    # Install dependencies
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    # Start Uvicorn server with hot reload
    subprocess.run(["uvicorn", "main:app", "--reload"])
    
    os.chdir("..")

# ---------------------- Run All ----------------------

if __name__ == "__main__":
    try:
        start_frontend()
        time.sleep(10)  # Optional: wait for frontend before starting backend
        start_backend()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error: {e}")
