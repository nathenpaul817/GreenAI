import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from tracker import UnifiedTracker


class ImageDataset(Dataset):
    """Custom dataset for loading images from ILSVRC_train folder"""
    
    def __init__(self, image_dir, transform=None, num_classes=10):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.num_classes = num_classes
        
        self.image_files = sorted([
            f for f in self.image_dir.glob("*.JPEG") 
            if f.is_file()
        ])
        
        if not self.image_files:
            raise ValueError(f"No JPEG images found in {image_dir}")
        
        print(f"Loaded {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Create pseudo-label based on image index (for demonstration)
            # In a real scenario, you would have actual class labels
            label = idx % self.num_classes
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image on error
            dummy_image = torch.zeros(3, 224, 224)
            label = 0
            return dummy_image, label



def get_device():
    """Get the appropriate device (TPU, CUDA, MPS, or CPU)"""
    # Try TPU (torch_xla)
    try:
        device = xm.xla_device()
        print("Using TPU via torch_xla")
        return device
    except ImportError:
        pass
    except Exception as e:
        print(f"torch_xla found but failed to initialize TPU: {e}")
    # Fallback to CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_resnet18(tool, batch_size, epochs, precision, image_dir="./ILSVRC_train"):
    """Train ResNet18 on ILSVRC images with energy tracking"""
    
    print(f"\n{'='*60}")
    print(f"ResNet18 Image Classifier Training")
    print(f"{'='*60}")
    print(f"Tool: {tool}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Precision: {precision}")
    print(f"{'='*60}\n")
    

    # Get device
    device = get_device()

    # Set precision
    dtype = torch.float16 if precision == "fp16" else torch.float32
    
    # Data preprocessing and augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = ImageDataset(
        image_dir=image_dir,
        transform=train_transforms,
        num_classes=10
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    # Load pre-trained ResNet18
    print("Loading ResNet18 model...")
    model = models.resnet18(pretrained=True)
    
    # Modify final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # 10 classes for demo
    

    # Move model to device and set dtype
    model = model.to(device)
    model = model.to(dtype)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize tracker
    tracker = UnifiedTracker(
        experiment_name="ResNet18_ImageClassifier",
        tool=tool
    )
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print(f"\nStarting training ({epochs} epochs)...")
    tracker.start()
    start_time = time.time()
    
    total_batches = len(dataloader)
    

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                # Move to device with correct dtype
                images = images.to(device).to(dtype)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # For TPU: mark step
                try:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                except ImportError:
                    pass

                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print progress
                if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{total_batches}] "
                          f"Loss: {loss.item():.4f}")

            # Epoch statistics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total

            print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        end_time = time.time()
        tracker.stop()
        runtime = end_time - start_time

        print(f"\nTraining completed in {runtime:.2f}s")

        # Get results
        results = tracker.get_results()
        results['runtime_seconds'] = runtime
        results['model'] = 'ResNet18'
        results['dataset_size'] = len(dataset)
        results['epochs'] = epochs
        results['batch_size'] = batch_size
        results['precision'] = precision

        print(f"\nEXPERIMENT_RESULTS: {json.dumps(results)}")

        return tracker, runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet18 image classifier on ILSVRC images"
    )
    parser.add_argument(
        "--tool",
        choices=["codecarbon", "eco2ai"],
        default="codecarbon",
        help="Energy tracking tool to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Training precision (fp32 or fp16)"
    )
    parser.add_argument(
        "--image-dir",
        default="./ILSVRC_train",
        help="Path to directory containing training images"
    )
    
    args = parser.parse_args()
    
    # Verify image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' not found!")
        exit(1)
    
    # Run training
    train_resnet18(
        tool=args.tool,
        batch_size=args.batch_size,
        epochs=args.epochs,
        precision=args.precision,
        image_dir=args.image_dir
    )
