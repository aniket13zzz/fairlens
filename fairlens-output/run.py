#!/usr/bin/env python3
"""
FairLens AI v2.0 — Quick Start Entry Point
Run: python run.py
"""
import os
import shutil
import subprocess
import sys


def main():
    print("""
╔══════════════════════════════════════════════╗
║         🔍 FairLens AI v2.0                  ║
║         Know Before You Deploy               ║
╚══════════════════════════════════════════════╝
    """)

    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("ℹ️  No .env found. Copying .env.example → .env")
        shutil.copy(".env.example", ".env")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Railway sets HOST=0.0.0.0 and PORT automatically via env vars
    # Locally: defaults to 127.0.0.1:8000
    is_railway = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID")
    host = os.getenv("HOST", "0.0.0.0" if is_railway else "127.0.0.1")
    port = os.getenv("PORT", "8000")

    print(f"🚀 Starting FairLens AI...")
    print(f"📊 App:      http://{host}:{port}/app")
    print(f"📖 API Docs: http://{host}:{port}/docs")
    print(f"🏥 Health:   http://{host}:{port}/health")
    print("\nPress Ctrl+C to stop\n")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", host,
        "--port", str(port),
    ])


if __name__ == "__main__":
    main()
