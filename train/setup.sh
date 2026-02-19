# Create environment
source slm/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets tokenizers wandb
pip install deepspeed flash-attn ninja