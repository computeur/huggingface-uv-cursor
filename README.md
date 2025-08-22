# huggingface-uv-cursor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![uv](https://img.shields.io/badge/uv-managed-6E56CF)
![macOS](https://img.shields.io/badge/MPS-Ready-black?logo=apple)
![Transformers](https://img.shields.io/badge/🤗_Transformers-enabled-yellow)
![JupyterLab](https://img.shields.io/badge/JupyterLab-ready-orange)
![Gradio](https://img.shields.io/badge/Gradio-demo-green)
![License](https://img.shields.io/badge/License-MIT-informational)

macOS/MPS‑optimized Hugging Face starter managed by uv; works in VS Code/Cursor. Jupyter, Accelerate, and Gradio included.

## 🚀 Quick Start

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the demo script:**
   ```bash
   python src/demo.py
   ```

3. **Start JupyterLab:**
   ```bash
   jupyter lab
   ```

## 📁 Project Structure

```
huggingface-uv-cursor/
├── .vscode/
│   └── settings.json          # VS Code/Cursor settings
├── notebooks/
│   └── intro.ipynb           # Jupyter notebook examples
├── src/
│   ├── demo.py               # Basic HF model demo
│   ├── training_example.py   # Accelerate training demo
│   └── gradio_demo.py        # Web interface demo
├── data/                     # Data files (gitignored if large)
├── .venv/                    # Virtual environment
├── accelerate_config.yaml    # Accelerate configuration
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project dependencies
└── README.md               # This file
```

## 🛠️ Setup

This project uses:
- **uv** for fast Python package management
- **PyTorch** with CPU wheels (MPS acceleration on Apple Silicon)
- **Transformers** for pre-trained models
- **Datasets** for data loading
- **Accelerate** for distributed training
- **JupyterLab** for interactive development
- **Gradio** for web demos
- **WandB** for experiment tracking

## 🍎 Apple Silicon (MPS) Support

The project is configured to use MPS acceleration when available:

```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## 🔧 VS Code/Cursor Configuration

The `.vscode/settings.json` file configures:
- Python interpreter pointing to `.venv/bin/python`
- Automatic virtual environment activation
- Jupyter notebook settings
- Type checking mode

## 📚 Examples

### Basic Text Generation
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")
pipe.model.to(device)  # Move to MPS if available
result = pipe("Hello world:", max_new_tokens=20)
```

### Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to(device)
```

### Training with Accelerate
```bash
python src/training_example.py
```

### Web Demo with Gradio
```bash
python src/gradio_demo.py
```

## 💡 Gradio tips

- **Pick a free port automatically**: we launch with `server_port=None`, so Gradio chooses an available port. If you prefer a fixed port:
  ```bash
  GRADIO_SERVER_PORT=7861 python src/gradio_demo.py
  ```
- **Local vs public link**:
  - Local only (default): opens at `http://127.0.0.1:<port>`.
  - Public share link:
    ```python
    # in src/gradio_demo.py
    demo.launch(share=True)
    ```
    Useful for quick demos; anyone with the link can access while the app runs.
- **Port already in use**: free 7860 and retry
  ```bash
  lsof -i :7860 | awk 'NR>1{print $2}' | xargs -r kill
  python src/gradio_demo.py
  ```

## 🚀 Next Steps

1. **Authenticate with Hugging Face:**
   ```bash
   huggingface-cli login
   ```

2. **Explore models and datasets:**
   - Visit [Hugging Face Hub](https://huggingface.co/)
   - Try different models in `notebooks/intro.ipynb`

3. **Create your own experiments:**
   - Add new scripts in `src/`
   - Create notebooks in `notebooks/`

## 📦 Dependencies

Core dependencies are defined in `pyproject.toml` and require Python 3.10+.

Install with uv (recommended):
```bash
uv pip install -e .
```

If you prefer pip:
```bash
pip install -e .
```

## 🔒 Privacy & safety

- No secrets are committed. Authenticate locally with `huggingface-cli login`; tokens are stored in your keychain and never written to this repo.
- Large artifacts and private data are ignored via `.gitignore` (`data/`, `datasets/`, model files, logs, caches).
- Before making this public, quickly scan your git history for accidental secrets:
  ```bash
  git log -p | grep -iE "(hf_|token|password|api|secret)" || true
  ```

## 🧩 Use this as a template

Click “Use this template” on GitHub or run:
```bash
git clone <your-repo-url> my-hf-sandbox
cd my-hf-sandbox && ./setup.sh
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python src/demo.py`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.
