#!/bin/bash
# Start VNC + DataWarrior with optimized resolution

set -e

export DISPLAY=:2
VNC_PORT=5902
NOVNC_PORT=6081
VNC_PASSWORD="datawarrior"

echo "🚀 Starting VNC stack..."

# Cleanup
pkill -f "Xvfb :2" 2>/dev/null || true
pkill -f "x11vnc.*5902" 2>/dev/null || true
pkill -f "websockify.*6081" 2>/dev/null || true
sleep 2

# Start Xvfb with optimized resolution
echo "📺 Starting Xvfb..."
Xvfb :2 -screen 0 2560x900x24 +extension RANDR &
sleep 2

# Create openbox config for auto-maximize and no decorations
mkdir -p ~/.config/openbox
cat > ~/.config/openbox/rc.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <theme>
    <titleLayout></titleLayout>
  </theme>
  <applications>
    <application class="*">
      <maximized>true</maximized>
      <decor>no</decor>
      <focus>yes</focus>
    </application>
  </applications>
</openbox_config>
EOF

# Start window manager with auto-maximize
echo "🖼️ Starting Openbox..."
DISPLAY=:2 openbox --config-file ~/.config/openbox/rc.xml &
sleep 1

# Start x11vnc
echo "🔗 Starting x11vnc..."
mkdir -p ~/.vnc
x11vnc -storepasswd $VNC_PASSWORD ~/.vnc/passwd 2>/dev/null || true
x11vnc -display :2 -forever -usepw -rfbport $VNC_PORT -shared -noxdamage &
sleep 2

# Start noVNC
echo "🌐 Starting noVNC..."
websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT &
sleep 2

echo ""
echo "✅ VNC Ready!"
echo "   📺 noVNC: http://localhost:$NOVNC_PORT/vnc.html"
echo "   🔒 Password: $VNC_PASSWORD"
echo "   📐 Resolution: 2560x900"
echo ""
