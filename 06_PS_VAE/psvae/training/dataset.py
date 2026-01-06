"""
Dataset classes for PS-VAE training.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List, Callable
import json


class ImageDataset(Dataset):
    """
    Simple image dataset for PS-VAE autoencoder training.
    
    Args:
        data_dir: Directory containing images
        image_size: Target image size
        transform: Optional custom transform
        extensions: Valid image extensions
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Collect image paths
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        self.image_paths.sort()
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image


class ImageTextDataset(Dataset):
    """
    Image-text dataset for text-to-image training.
    
    Expects either:
    1. A directory with images and a metadata.json file containing captions
    2. A directory with images where each image has a corresponding .txt file
    
    Args:
        data_dir: Directory containing images and captions
        image_size: Target image size
        tokenizer: Text tokenizer (e.g., T5 tokenizer)
        max_text_length: Maximum text token length
        transform: Optional custom transform
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        tokenizer: Optional[Callable] = None,
        max_text_length: int = 77,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load data
        self.samples = self._load_samples()
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
    
    def _load_samples(self) -> List[dict]:
        """Load image-caption pairs."""
        samples = []
        
        # Check for metadata.json
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            for item in metadata:
                image_path = os.path.join(self.data_dir, item["image"])
                if os.path.exists(image_path):
                    samples.append({
                        "image_path": image_path,
                        "caption": item.get("caption", ""),
                    })
        else:
            # Look for .txt files alongside images
            extensions = (".jpg", ".jpeg", ".png", ".webp")
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(extensions):
                        image_path = os.path.join(root, file)
                        txt_path = os.path.splitext(image_path)[0] + ".txt"
                        
                        caption = ""
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as f:
                                caption = f.read().strip()
                        
                        samples.append({
                            "image_path": image_path,
                            "caption": caption,
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption = sample["caption"]
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            # Return dummy tensors if no tokenizer (for testing without text encoder)
            input_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_length, dtype=torch.long)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
        }


class LatentDataset(Dataset):
    """
    Dataset of pre-computed latents for efficient DiT training.
    
    Pre-computing PS-VAE latents speeds up DiT training significantly.
    
    Args:
        latent_dir: Directory containing .pt files with latents
        text_embed_dir: Directory containing .pt files with text embeddings
    """
    
    def __init__(
        self,
        latent_dir: str,
        text_embed_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.latent_dir = latent_dir
        self.text_embed_dir = text_embed_dir
        
        # Collect latent files
        self.latent_files = sorted([
            f for f in os.listdir(latent_dir) if f.endswith(".pt")
        ])
    
    def __len__(self) -> int:
        return len(self.latent_files)
    
    def __getitem__(self, idx: int) -> dict:
        latent_file = self.latent_files[idx]
        latent_path = os.path.join(self.latent_dir, latent_file)
        
        # Load latent
        latent = torch.load(latent_path, map_location="cpu")
        
        result = {"latent": latent}
        
        # Load text embedding if available
        if self.text_embed_dir is not None:
            text_file = latent_file.replace(".pt", "_text.pt")
            text_path = os.path.join(self.text_embed_dir, text_file)
            if os.path.exists(text_path):
                text_data = torch.load(text_path, map_location="cpu")
                result["text_embed"] = text_data["embed"]
                result["text_mask"] = text_data.get("mask", None)
        
        return result

