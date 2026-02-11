#!/bin/bash
# Starts everything: VNC + Streamlit

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════╗"
echo "║    🔬 DataWarrior-Agent v1 -Start      ║"
echo "╚════════════════════════════════════════╝"

# Create logs dir
mkdir -p logs

# Start VNC
echo "📺 Starting VNC..."
./scripts/start_vnc.sh > logs/vnc.log 2>&1 &
sleep 5

# Start Streamlit
echo "🌐 Starting Streamlit..."
streamlit run streamlit_app.py --server.port 8501 --server.headless true > logs/streamlit.log 2>&1 &
sleep 3

echo ""
echo "╔════════════════════════════════════════╗"
echo "║           ✅ Ready!                    ║"
echo "╠════════════════════════════════════════╣"
echo "║  🌐 App:    http://localhost:8501      ║"
echo "║  📺 VNC:    http://localhost:6081      ║"
echo "║  🔌 TCP:    5151                       ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "📝 Logs: logs/vnc.log, logs/streamlit.log"
echo "🛑 Stop: ./scripts/stop_all.sh"
