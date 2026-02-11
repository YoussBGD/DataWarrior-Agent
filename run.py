#!/usr/bin/env python3
"""
DataWarrior-AI v6 Launcher
Runs scripts in order: stop -> vnc -> start
Then opens http://localhost:8501 in the browser
"""

import subprocess
import os
import time
import webbrowser



def main():
    print("=" * 50)
    print("   DataWarrior-AI - Launcher")
    print("=" * 50)
    print()
    
    # 1. Stop all
    os.system("./scripts/stop_all.sh")
    
    # 2. Start VNC
    os.system("./scripts/start_vnc.sh")

    
    # 3. Start all
    os.system("./scripts/start_all.sh")

    
    # 4. Open browser
    url = "http://localhost:8501"
    print(f"\n🌐 Opening {url} in browser...")
    webbrowser.open(url)
    
    print("\n✨ Launch complete!")
    print(f"   Access the application: {url}")

if __name__ == "__main__":
    main()
