#  DataWarrior-Agent

**Natural Language Control of Computational Chemistry Software « DataWarrior » Through Macro Orchestration** 
https://chemrxiv.org/doi/full/10.26434/chemrxiv.10001959/v1

DataWarrior-Agent enables users to control [DataWarrior](https://openmolecules.org/datawarrior/), a powerful cheminformatics platform, using natural language. Instead of navigating complex menus, simply describe what you want to do in plain English (or French), and the AI assistant will execute the appropriate operations. you can also ask questions about your dataset. 

---

##  Features

- **Natural Language Interface** - Control DataWarrior by typing commands like "calculate logP and molecular weight" or "apply Lipinski filter"
- **Multiple LLM Support** - Works with OpenAI (GPT-4), Anthropic (Claude), or local models via Ollama
- **Privacy-Friendly** - Run entirely locally with Ollama, no data sent to external servers
- **Real-time Visualization** - See DataWarrior's interface live through embedded VNC viewer
- **Easy File Upload** - Drag and drop SDF, MOL, CSV files directly

---

##  Current Capabilities

This tool is under active development. Here are the currently implemented features:

### File Operations
| Command | Description |
|---------|-------------|
| Open files | Load molecular files (SDF, MOL, CSV with SMILES, TXT) |
| Export to CSV | Save the current dataset to CSV format |
| Auto-save snapshots | Automatic snapshots for data consultation |

### Molecular Descriptor Calculation
| Command | Description |
|---------|-------------|
| Calculate all descriptors | Compute all 59 molecular descriptors available in DataWarrior |
| Calculate specific descriptors | Compute selected properties from the following categories: |

**Available descriptor categories:**
- **Weight:** totalWeight, fragmentWeight, fragmentAbsWeight
- **Drug-likeness:** logP, logS, acceptors, donors, sasa, rpsa, tpsa, druglikeness
- **Toxicity:** mutagenic, tumorigenic, reproEffective, irritant, nasty, pains
- **Structure:** shape, flexibility, complexity, fragments
- **Atoms:** heavyAtoms, nonCHAtoms, metalAtoms, negAtoms, stereoCenters, aromAtoms, sp3CFraction, sp3Atoms, symmetricAtoms
- **Bonds:** nonHBonds, rotBonds, closures
- **Rings:** largestRing, rings, carbo, heteroRings, satRings, nonAromRings, aromRings, and more
- **Functional groups:** amides, amines, alkylAmines, arylAmines, aromN, basicN, acidicO
- **Stereochemistry:** stereoConfiguration
- **3D properties:** globularity, globularity2, surface3d, volume3d

### Filtering
| Command | Description |
|---------|-------------|
| Lipinski filter | Apply Rule of Five to identify drug-like molecules (MW≤500, logP≤5, HBD≤5, HBA≤10) |

### Data Manipulation
| Command | Description |
|---------|-------------|
| Delete columns | Remove unwanted columns from the dataset |
| Generate 2D coordinates | Regenerate 2D coordinates to align molecules by common scaffold |

### Data Consultation
| Command | Description |
|---------|-------------|
| Read dataset | Load current data for analysis |
| Natural language queries | Ask questions about data content (columns, values, statistics) |

> **Note:** Query performance depends heavily on the LLM used. GPT-4/GPT-4o give good results, while smaller models like LLaMA 3.1 8B may struggle with complex statistics.

### Session Management
| Command | Description |
|---------|-------------|
| Connection status | Check TCP connection to DataWarrior |
| Session info | View current session state and history |
| List tools | Display all available tools |

---

## Extensibility

**This tool is designed to be extended!** We are actively adding new macros and features.

You can also add your own tools by following the step-by-step guide in the technical documentation:

 **[TECHNICAL_DOCUMENTATION_DataWarrior-AI.pdf](1-files-documentation/TECHNICAL_DOCUMENTATION_DataWarrior-AI-EN.pdf)** → Section "Guide: Adding a New Feature"

The guide covers:
1. Creating macros in DataWarrior (recording and exporting)
2. Modifying exported macros for dynamic parameters
3. Adding methods in `MacroModifier`
4. Creating MCP tools in `mcp_server.py`
5. Updating configuration files (descriptions and schemas)
6. Integrating with the Streamlit interface
7. Testing your new functionality

**Contributions are welcome!** If you create useful new tools, consider submitting a pull request.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Step 1: Install DataWarrior](#step-1-install-datawarrior)
   - [Step 2: Install the TCP Plugin](#step-2-install-the-tcp-plugin)
   - [Step 3: Install Ollama (Optional)](#step-3-install-ollama-optional)
   - [Step 4: Set Up Python Environment](#step-4-set-up-python-environment)
   - [Step 5: Install System Dependencies](#step-5-install-system-dependencies)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)
6. [Architecture](#architecture)

---

## Prerequisites

Before starting, ensure you have:

| Requirement | Version | Notes |
|-------------|---------|-------|
| Operating System | Linux (Ubuntu 20.04+) | Windows/macOS: partial support |
| Python | 3.11+ | Required |
| Conda | Any recent version | Miniconda or Anaconda |
| RAM | 16 GB+ | 8 GB minimum, 16+ recommended for local LLMs |
| Disk Space | ~10 GB | For models and dependencies |

---

## Installation

### Step 1: Install DataWarrior

DataWarrior is the core cheminformatics platform that this tool controls.

#### Linux (Debian/Ubuntu)

```bash
# Download the latest version
wget https://openmolecules.org/datawarrior/datawarrior_linux6.1.zip

# Extract
unzip datawarrior_linux6.1.zip -d ~/

# Make executable
chmod +x ~/datawarrior/datawarrior

# (Optional) Create desktop shortcut or add to PATH
echo 'export PATH="$HOME/datawarrior:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Alternative: Download from website

1. Go to [https://openmolecules.org/datawarrior/download.html](https://openmolecules.org/datawarrior/download.html)
2. Download the version for your operating system
3. Follow the installation instructions for your OS

#### Verify Installation

```bash
# Launch DataWarrior to verify it works
datawarrior
# or
~/datawarrior/datawarrior
```

You should see the DataWarrior application window open.

---

### Step 2: Install the TCP Plugin

The TCP plugin enables external communication with DataWarrior. Without it, the AI cannot send commands to DataWarrior.

#### Locate Your Plugin Directory

| OS | Plugin Directory |
|----|------------------|
| **Linux** | `~/.datawarrior/plugin/` |
| **Windows** | `%APPDATA%\DataWarrior\plugin\` |
| **macOS** | `~/Library/Application Support/DataWarrior/plugin/` |

#### Install the Plugin

# Copy the plugin JAR file (from this repository)
cp plugin/macroagent.jar ~/.datawarrior/plugin/
```

#### Verify Plugin Installation

1. **Restart DataWarrior** (close and reopen if it was running)
2. Go to menu: **Tools** → You should see **"Start Macro AI Agent"**
3. Click it to activate the TCP server (port 5151)

> **Note:** The plugin will be automatically activated when you upload a file through the Streamlit interface. Manual activation is only needed for testing.

---

### Step 3: Install Ollama 

**Already have Ollama and llama3.1:8b installed?** Skip to [Step 4](#step-4-set-up-python-environment).


Ollama allows you to run LLMs locally for complete privacy. **Skip this step** if you only plan to use cloud APIs (OpenAI/Anthropic).

#### Install Ollama

```bash
# Linux - One-line installer
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

#### Download LLaMA 3.1 8B Model

```bash
# Pull the model (~4.7 GB download)
ollama pull llama3.1:8b

# Verify the model is available
ollama list
```

#### Test Ollama

```bash
# Quick test
ollama run llama3.1:8b "Hello, how are you?"

# Check if Ollama server is running
curl http://localhost:11434/api/tags
```

> **Hardware Notes:**
> - **CPU only (16GB RAM):** LLaMA 3.1 8B works but responses take at least 30-60 seconds depending on your query
> - **GPU (8GB+ VRAM):** Responses in 2-5 seconds
> - **For larger models** (70B), you need 24GB+ VRAM


---

### Step 4: Set Up Python Environment

We recommend using Conda to manage the Python environment.

#### Create Conda Environment

```bash
# Create a new environment with Python 3.11
conda create -n DataWarrior-ai python=3.11 -y

# Activate the environment
conda activate DataWarrior-ai
```

#### Clone the Repository

```bash
# Clone DataWarrior-Agent
git clone https://github.com/yourusername/DataWarrior-Agent.git
cd DataWarrior-Agent
```

#### Install Python Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

> **Note:** This will install ~3 GB of dependencies including PyTorch and Transformers. The first installation may take 5-10 minutes depending on your internet connection.

#### Verify Installation

```bash
# Check that key packages are installed
python -c "import streamlit; import mcp; import anthropic; import openai; print('✅ All packages installed successfully!')"
```

---

### Step 5: Install System Dependencies

The VNC stack is required to display DataWarrior's interface in the browser.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y xvfb x11vnc novnc websockify openbox
```

#### Verify VNC Installation

```bash
# Check that all components are installed
which Xvfb x11vnc websockify
```

---

## Configuration

### API Keys (for Cloud LLMs)

If you want to use OpenAI or Anthropic models, you'll need API keys.

#### Option 1: Environment Variables

```bash
# Add to your ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
```

#### Option 2: Enter in the Interface

You can also enter API keys directly in the Streamlit sidebar. They will be saved locally in `.config/api_keys.json`.

### Project Structure

After installation, your directory should look like this:

```
DataWarrior-Agent/
├── config/
│   ├── descriptors_catalog.json
│   ├── macro_descriptions.json
│   └── macro_schemas.json
├── core/
│   ├── __init__.py
│   ├── macro_modifier.py
│   └── tcp_client.py
├── data/
│   ├── input/
│   └── output/
├── llm/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── orchestrator.py
│   ├── semantic_search.py
│   └── token_manager.py
├── macros/
│   ├── open.dwam
│   ├── save_csv_file.dwam
│   ├── calcul_selective_descriptors.dwam
│   ├── ...
│   └── ...
├── plugin/
│   └── macroagent.jar
├── scripts/
│   ├── start_all.sh
│   ├── start_vnc.sh
│   └── stop_all.sh
├── 1-files-documentation/
│   └── TECHNICAL_DOCUMENTATION_DataWarrior-AI-EN.pdf.pdf
├── mcp_server.py
├── streamlit_app.py
├── run.py
├── requirements.txt
└── README.md
```

---

## Usage

### Quick Start

```bash
# Make sure you're in the project directory with conda env activated
conda activate DataWarrior-ai
cd DataWarrior-ai

# Launch the application
python run.py
```

This will:
1. Start the VNC server
2. Launch the Streamlit interface
3. Open your browser to `http://localhost:8501`

### Alternative: Manual Start

```bash
# Start VNC stack
./scripts/start_vnc.sh

# In another terminal, start Streamlit
streamlit run streamlit_app.py --server.port 8501
```

### Using the Interface

1. **Select an LLM** in the sidebar:
   - **OpenAI:** Enter your API key and select a model (GPT-4, GPT-4o-mini)
   - **Anthropic:** Enter your API key and select a model (Claude 3.5 Sonnet)
   - **Ollama:** Select a local model (requires Ollama running)

2. **Upload a molecular file:**
   - Drag and drop or click to upload SDF, MOL, or CSV files
   - The file will automatically open in DataWarrior

3. **Chat with the AI:**
   - Type natural language commands like:
     - *"Calculate logP, molecular weight, and TPSA"*
     - *"Apply Lipinski Rule of Five filter"*
     - *"How many molecules have logP greater than 3?"*
     - *"Save the results as filtered_molecules.csv"*

4. **Watch DataWarrior:**
   - See the operations executed in real-time in the VNC viewer

### Stopping the Application

```bash
./scripts/stop_all.sh
```

---

## Troubleshooting

### DataWarrior TCP Plugin Not Connected

**Symptom:** Red "❌ Not connected" in sidebar

**Solutions:**
1. Ensure DataWarrior is running
2. Manually activate: **Tools** → **Start Macro AI Agent**
3. Check if port 5151 is available: `netstat -tlnp | grep 5151`

### Ollama Connection Failed

**Symptom:** Error when selecting Ollama model

**Solutions:**
1. Check if Ollama is running: `curl http://localhost:11434/api/tags`
2. Start Ollama service: `ollama serve`
3. Verify model is downloaded: `ollama list`

### VNC Not Displaying

**Symptom:** Black or empty VNC viewer

**Solutions:**
1. Check VNC logs: `cat logs/vnc.log`
2. Restart VNC stack: `./scripts/stop_all.sh && ./scripts/start_vnc.sh`
3. Verify Xvfb is running: `ps aux | grep Xvfb`

### Slow Local Model Responses

**Symptom:** 30+ seconds for Ollama responses

**Solutions:**
- This is normal for CPU-only inference
- Use a GPU for faster responses
- Try smaller models: `ollama pull llama3.2:3b`

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│   User Chat     │────▶│  LLM Orchestrator│────▶│   MCP Server    │
│   (Streamlit)   │     │  (GPT/Claude/    │     │   (Python)      │
│                 │     │   Ollama)        │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          │ TCP :5151
                                                                                      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│   VNC Viewer    │◀────│   DataWarrior    │◀────│  TCP Plugin     │
│   (noVNC)       │     │   (Java App)     │     │  (macroagent)   │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Streamlit App** | Web interface with chat and VNC viewer |
| **LLM Orchestrator** | Two-stage (cloud) or single-stage (local) workflow |
| **MCP Server** | Exposes DataWarrior tools via Model Context Protocol |
| **TCP Plugin** | Java plugin enabling external control of DataWarrior |
| **Macro Modifier** | Dynamically updates .dwam macro files |

### LLM Strategies

| Model Type | Strategy | Best For |
|------------|----------|----------|
| **GPT, Claude** | Two-Stage (Planning → Parameterization) | Accuracy, complex workflows |
| **Ollama (LLaMA, Mistral)** | Semantic Search + Single-Stage | Privacy, cost savings | Less efficient |

---

## License

[MIT License](LICENSE)

## Citation

If you use DataWarrior-Agent in your research, please cite:

```bibtex
@article{DataWarrior-Agent2025,
  title={Natural Language Control of DataWarrior through Macro Orchestration},
  author={Bagdad, Youcef and Villoutreix, Bruno},
  year={2025}
}
```

## Acknowledgments

- [DataWarrior](https://openmolecules.org/datawarrior/) by Thomas Sander (Idorsia)
- [Anthropic](https://www.anthropic.com/) for the Model Context Protocol
- INSERM UMR 1141 and SATT Aquitaine for funding support
