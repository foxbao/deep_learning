# import model_loader
# import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

def main():
    DEVICE = "cpu"

    ALLOW_CUDA = True
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

if __name__ == '__main__':
    main()
