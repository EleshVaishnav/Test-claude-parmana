#!/usr/bin/env bash
set -e

echo "▸ Parmana 2.0 installer"

# Python check
if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

PY_VER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_VER" -lt 10 ]; then
  echo "✗ Python 3.10+ required. Found 3.$PY_VER"
  exit 1
fi

# Clone or update
if [ -d "Parmana-2.0" ]; then
  echo "▸ Directory exists — pulling latest..."
  cd Parmana-2.0 && git pull
else
  git clone https://github.com/EleshVaishnav/Parmana-2.0.git
  cd Parmana-2.0
fi

# Virtual environment
if [ ! -d ".venv" ]; then
  echo "▸ Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip silently
pip install --upgrade pip -q

# Install dependencies
echo "▸ Installing dependencies (this may take a minute)..."
pip install -r requirements.txt -q

# .env setup
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "▸ .env created from .env.example — add your API keys."
else
  echo "▸ .env already exists — skipping."
fi

echo ""
echo "✓ Done. To start:"
echo "  cd Parmana-2.0 && source .venv/bin/activate && python main.py"
