#!/usr/bin/env zsh
set -euo pipefail

echo "🚀 Setting up Hugging Face Sandbox..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install dependencies
echo "📥 Installing PyTorch (CPU + MPS)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "📥 Installing Hugging Face stack..."
uv pip install transformers datasets accelerate huggingface_hub jupyterlab ipywidgets

echo "📥 Installing additional tools..."
uv pip install evaluate gradio wandb

# Create project structure
echo "📁 Creating project structure..."
mkdir -p .vscode notebooks src data

# Create VS Code settings
echo "⚙️ Configuring VS Code settings..."
cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "jupyter.askForKernelRestart": false,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.analysis.typeCheckingMode": "basic",
  "terminal.integrated.inheritEnv": true
}
JSON

# Create accelerate config
echo "⚙️ Creating Accelerate configuration..."
cat > accelerate_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: ''
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
use_mps_device: true
YAML

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    echo ".venv/" >> .gitignore
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Open this folder in Cursor/VS Code"
echo "2. Select the Python interpreter: .venv/bin/python"
echo "3. Run: python src/demo.py"
echo "4. Start JupyterLab: jupyter lab"
echo "5. Authenticate with HF: huggingface-cli login"
echo ""
echo "🚀 Happy coding with Hugging Face!"
