#!/bin/bash
# Stops everything

echo "🛑 Stopping DataWarrior-Agent..."

pkill -f "streamlit run" 2>/dev/null || true
pkill -f "Xvfb :2" 2>/dev/null || true
pkill -f "x11vnc.*5902" 2>/dev/null || true
pkill -f "websockify.*6081" 2>/dev/null || true
pkill -f "openbox" 2>/dev/null || true
pkill -f datawarrior 2>/dev/null || true

echo "✅ Stopped"
