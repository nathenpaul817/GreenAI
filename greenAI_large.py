import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config, GPT2LMHeadModel
from torchvision import models
import argparse

from tracker_utils import UnifiedTracker

def get_model(model_name):
    if model_name == "resnet18":
        # Tier II: Computer Vision
        return models.resnet18(num_classes=10)
    elif model_name == "gpt2":
        # Tier III: NLP (Small scale config for demo)
        config = GPT2Config(n_layer=6, n_head=6) 
        return GPT2LMHeadModel(config)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(args.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Precision Setup (FP32 vs FP16)
    use_amp = (args.precision == "fp16")
    scaler = GradScaler(enabled=use_amp) # Scaler handles gradient unscaling for FP16
    
    # replace with DataLoader(ImageNet or WikiText)
    input_shape = (args.batch_size, 3, 224, 224) if args.model == "resnet18" else (args.batch_size, 128)
    dummy_input = torch.randn(input_shape).to(device)
    dummy_target = torch.randint(0, 10, (args.batch_size,)).to(device)
    
    tracker = UnifiedTracker(
        experiment_name=f"{args.model}_{args.precision}_bs{args.batch_size}", 
        tool="codecarbon"
    )
    
    print(f"Starting {args.model} training | Precision: {args.precision} | Batch: {args.batch_size}")
    tracker.start()
    
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Mixed Precision Context Manager
        with autocast(enabled=use_amp):
            if args.model == "gpt2":
                outputs = model(input_ids=dummy_input.long(), labels=dummy_input.long())
                loss = outputs.loss
            else:
                outputs = model(dummy_input)
                loss = criterion(outputs, dummy_target)

        # Backward pass with scaling if FP16
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    tracker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "gpt2"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train(args)