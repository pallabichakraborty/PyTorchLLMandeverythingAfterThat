# All the boring stuff

## PyTorchLLM and everything After That

# Environment Setup

brew install uv
uv --version
uv venv
source .venv/bin/activate && uv pip install torch torchvision torchaudio
source .venv/bin/activate && python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS (Apple Silicon) available: {torch.backends.mps.is_available()}')"

# To use Environment

## Activate the virtual environment
source .venv/bin/activate

## Run your Python scripts
python simple_nueral_network.py

## Deactivate when done
deactivate


# Installing additional packages:

## With the environment activated
uv pip install <package-name>

## Or without activation
uv pip install --python .venv/bin/python <package-name>


If using Apple Silicon, use MPS for GPU-accelerated training by using device = torch.device("mps") in the code.

# Other package installation
uv pip install matplotlib
