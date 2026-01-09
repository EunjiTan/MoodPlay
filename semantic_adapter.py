"""
Semantic Adapter for Grayscale-to-Color Image Generation
DISTRIBUTED TRAINING READY - Multiple people can train on different datasets
ControlNet-style conditioning adapter for Stable Diffusion
Phase 1: Semantic Control Adapter (Image → Segmentation → Color)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import hashlib
from datetime import datetime

# ============================================================================
# DISTRIBUTED TRAINING UTILITIES
# ============================================================================

class CheckpointManager:
    """
    Manages checkpoints for distributed training
    Handles merging models from multiple team members
    """
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.merge_dir = self.save_dir / "merged"
        self.merge_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, adapter, optimizer, epoch, step, loss, person_id):
        """
        Save checkpoint with person identifier
        person_id: unique identifier for each team member (e.g., 'alice', 'bob')
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'adapter_state_dict': adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'person_id': person_id,
            'timestamp': datetime.now().isoformat(),
            'num_params': sum(p.numel() for p in adapter.parameters()),
        }
        
        filename = f"adapter_{person_id}_epoch{epoch}_step{step}.pt"
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        
        # Also save a "latest" version for easy loading
        latest_path = self.save_dir / f"adapter_{person_id}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"✓ Checkpoint saved: {path}")
        return path
    
    def merge_checkpoints(self, checkpoint_paths, output_name="merged_adapter"):
        """
        Merge multiple checkpoints by averaging weights
        This is the key function for collaborative training!
        
        Args:
            checkpoint_paths: list of checkpoint file paths
            output_name: name for merged checkpoint
        """
        if len(checkpoint_paths) == 0:
            print("⚠ No checkpoints to merge")
            return None
        
        print(f"\n{'='*60}")
        print(f"MERGING {len(checkpoint_paths)} CHECKPOINTS")
        print(f"{'='*60}")
        
        # Load all checkpoints
        checkpoints = []
        for path in checkpoint_paths:
            try:
                ckpt = torch.load(path, map_location='cpu')
                checkpoints.append(ckpt)
                print(f"✓ Loaded: {Path(path).name}")
                print(f"  Person: {ckpt['person_id']}")
                print(f"  Steps: {ckpt['global_step']}")
                print(f"  Loss: {ckpt['loss']:.4f}")
            except Exception as e:
                print(f"✗ Failed to load {path}: {e}")
        
        if len(checkpoints) == 0:
            print("⚠ No valid checkpoints loaded")
            return None
        
        # Average the weights
        print("\nAveraging weights...")
        merged_state = {}
        
        # Get keys from first checkpoint
        keys = checkpoints[0]['adapter_state_dict'].keys()
        
        for key in keys:
            # Stack all weights for this key
            weights = torch.stack([
                ckpt['adapter_state_dict'][key].float() 
                for ckpt in checkpoints
            ])
            # Average
            merged_state[key] = weights.mean(dim=0)
        
        # Create merged checkpoint
        merged_checkpoint = {
            'adapter_state_dict': merged_state,
            'merged_from': [ckpt['person_id'] for ckpt in checkpoints],
            'num_merged': len(checkpoints),
            'timestamp': datetime.now().isoformat(),
            'individual_losses': [ckpt['loss'] for ckpt in checkpoints],
            'avg_loss': np.mean([ckpt['loss'] for ckpt in checkpoints]),
        }
        
        # Save merged checkpoint
        merged_path = self.merge_dir / f"{output_name}.pt"
        torch.save(merged_checkpoint, merged_path)
        
        print(f"\n✓ MERGED CHECKPOINT SAVED: {merged_path}")
        print(f"  Combined from: {merged_checkpoint['merged_from']}")
        print(f"  Average loss: {merged_checkpoint['avg_loss']:.4f}")
        print(f"{'='*60}\n")
        
        return merged_path


# ============================================================================
# PHASE 3: SEMANTIC ADAPTER ARCHITECTURE
# ============================================================================

class SemanticAdapter(nn.Module):
    """
    Lightweight CNN adapter that processes segmentation maps
    and produces features for injection into Stable Diffusion U-Net
    """
    def __init__(self, in_channels=1, base_channels=64, out_channels=320):
        super().__init__()
        
        # Encoder blocks (gradually increase channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Output projection to match U-Net channels
        self.proj = nn.Conv2d(base_channels*4, out_channels, 1)
        
        # Zero initialization for gradual learning
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, seg):
        """
        Args:
            seg: [B, 1, H, W] segmentation map (normalized 0-1)
        Returns:
            features: [B, out_channels, H//4, W//4] semantic features
        """
        x = self.conv1(seg)      # [B, 64, H, W]
        x = self.conv2(x)        # [B, 128, H//2, W//2]
        x = self.conv3(x)        # [B, 256, H//4, W//4]
        x = self.conv4(x)        # [B, 256, H//4, W//4]
        features = self.proj(x)  # [B, 320, H//4, W//4]
        return features


# ============================================================================
# PHASE 2: DATASET PREPARATION
# ============================================================================

class SemanticColorDataset(Dataset):
    """
    Dataset for semantic-guided colorization
    Returns: (grayscale_image, segmentation_map, target_color_image)
    """
    def __init__(self, gray_dir, seg_dir, color_dir, size=512):
        self.gray_dir = Path(gray_dir)
        self.seg_dir = Path(seg_dir)
        self.color_dir = Path(color_dir)
        self.size = size
        
        # Get sorted file lists
        self.gray_files = sorted([f.name for f in self.gray_dir.glob("*.png")] + 
                                 [f.name for f in self.gray_dir.glob("*.jpg")])
        
        assert len(self.gray_files) > 0, f"No images found in {gray_dir}"
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])
        
        self.norm = T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    
    def __len__(self):
        return len(self.gray_files)
    
    def __getitem__(self, idx):
        name = self.gray_files[idx]
        
        # Load images
        gray = Image.open(self.gray_dir / name).convert("L")
        seg = Image.open(self.seg_dir / name).convert("L")
        color = Image.open(self.color_dir / name).convert("RGB")
        
        # Transform
        gray = self.transform(gray)    # [1, H, W]
        seg = self.transform(seg)      # [1, H, W]
        color = self.transform(color)  # [3, H, W]
        
        # Normalize
        gray = self.norm(gray)
        color = self.norm(color)
        # Keep seg in [0, 1] range
        
        return {
            'gray': gray,
            'seg': seg,
            'color': color,
            'filename': name
        }


# ============================================================================
# PHASE 2: PREPROCESSING UTILITIES
# ============================================================================

def convert_to_grayscale(input_dir, output_dir):
    """Convert color images to grayscale"""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for img_file in tqdm(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")),
                         desc="Converting to grayscale"):
        img = Image.open(img_file).convert("L")
        img.save(output_path / img_file.name)
    print(f"✓ Converted {len(list(output_path.glob('*')))} images to grayscale")


def generate_segmentation_maps(input_dir, output_dir, model_name="yolov8x-seg.pt"):
    """Generate segmentation maps using YOLOv8"""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("⚠ ultralytics not installed. Run: pip install ultralytics")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_name)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for img_file in tqdm(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")),
                         desc="Generating segmentation maps"):
        results = model(str(img_file), verbose=False)
        
        if results[0].masks is not None:
            # Combine all masks into single binary map
            masks = results[0].masks.data.cpu().numpy()
            combined = np.any(masks, axis=0).astype(np.uint8) * 255
        else:
            # If no objects detected, create blank mask
            img = cv2.imread(str(img_file))
            combined = np.zeros(img.shape[:2], dtype=np.uint8)
        
        cv2.imwrite(str(output_path / img_file.name), combined)
    print(f"✓ Generated {len(list(output_path.glob('*')))} segmentation maps")


def generate_hybrid_segmentation_maps(input_dir, output_dir):
    """
    Combine YOLO (objects) + Semantic Segmentation (scenes)
    Detects: sky, buildings, roads, people, vehicles, vegetation, etc.
    """
    try:
        from ultralytics import YOLO
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        import cv2
        import torch
    except ImportError:
        print("⚠ Missing packages. Run: pip install ultralytics transformers")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models (this may take a moment)...")
    # Load both models
    yolo = YOLO("yolov8x-seg.pt")
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    semantic_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    print("✓ Models loaded")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for img_file in tqdm(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")),
                         desc="Generating hybrid segmentation"):
        
        # Get semantic segmentation (sky, buildings, etc.)
        image = Image.open(img_file).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = semantic_model(**inputs)
        semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()
        
        # Get object segmentation (people, cars, etc.)
        yolo_results = yolo(str(img_file), verbose=False)
        
        # Combine both (semantic scene + detected objects)
        combined = semantic_map.copy()
        if yolo_results[0].masks is not None:
            object_masks = yolo_results[0].masks.data.cpu().numpy()
            for mask in object_masks:
                # Resize mask to match semantic_map size if needed
                if mask.shape != combined.shape:
                    mask = cv2.resize(mask, (combined.shape[1], combined.shape[0]))
                combined = np.maximum(combined, mask * 255)
        
        # Normalize to 0-255 range
        combined = (combined * 255.0 / combined.max()).astype(np.uint8) if combined.max() > 0 else combined.astype(np.uint8)
        
        # Save
        cv2.imwrite(str(output_path / img_file.name), combined)
    
    print(f"✓ Generated {len(list(output_path.glob('*')))} hybrid segmentation maps")


# ============================================================================
# PHASE 4 & 5: TRAINING SYSTEM (DISTRIBUTED VERSION)
# ============================================================================

class SemanticAdapterTrainer:
    """
    Training system for Semantic Adapter with frozen Stable Diffusion
    DISTRIBUTED TRAINING READY
    """
    def __init__(self, 
                 adapter,
                 dataset,
                 person_id,  # NEW: identifier for this team member
                 lr=1e-5,
                 batch_size=4,
                 device='cuda'):
        
        self.adapter = adapter.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.person_id = person_id  # Store person ID
        
        # DataLoader
        self.loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            adapter.parameters(), 
            lr=lr,
            weight_decay=0.01
        )
        
        # Checkpoint manager
        self.ckpt_manager = CheckpointManager()
        
        # Load Stable Diffusion components (frozen)
        print("Loading Stable Diffusion components...")
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16
        ).to(device)
        
        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        # Freeze diffusion components
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()
        
        print("✓ Stable Diffusion loaded (frozen)")
        
        # Training state
        self.global_step = 0
        self.losses = []
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint (can be from any team member or merged)"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        if 'merged_from' in checkpoint:
            print(f"  This is a MERGED checkpoint from: {checkpoint['merged_from']}")
    
    @torch.no_grad()
    def encode_images(self, images):
        """Encode images to latent space"""
        latents = self.vae.encode(images.half()).latent_dist.sample()
        latents = latents * 0.18215  # SD scaling factor
        return latents
    
    def inject_semantic_features(self, noisy_latents, semantic_feat):
        """
        Simple feature injection (Phase 1 approach)
        Resize semantic features to match latent dimensions and add
        """
        # Resize to latent dimensions [B, 320, H//4, W//4] -> [B, 4, H//8, W//8]
        B, C, H, W = noisy_latents.shape
        feat_resized = F.interpolate(
            semantic_feat, 
            size=(H, W), 
            mode='bilinear',
            align_corners=False
        )
        
        # Project to latent channels (4) and add
        feat_projected = feat_resized[:, :4, :, :]  # Take first 4 channels
        conditioned_latents = noisy_latents + 0.1 * feat_projected.half()
        
        return conditioned_latents
    
    def train_step(self, batch):
        """Single training step"""
        gray = batch['gray'].to(self.device)
        seg = batch['seg'].to(self.device)
        color = batch['color'].to(self.device)
        
        # Encode target colors to latents
        with torch.no_grad():
            latents = self.encode_images(color)
        
        # Sample random timesteps
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (bsz,), device=self.device
        ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Generate semantic features (adapter forward pass)
        semantic_feat = self.adapter(seg)
        
        # Inject semantic conditioning
        conditioned_latents = self.inject_semantic_features(noisy_latents, semantic_feat)
        
        # Predict noise with U-Net
        # Use unconditional (empty prompt) for now
        with torch.no_grad():
            encoder_hidden_states = torch.zeros(
                bsz, 77, 768,
                device=self.device,
                dtype=torch.float16
            )
        
        noise_pred = self.unet(
            conditioned_latents.half(),
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred.float(), noise.float())
        
        # Backward pass (only adapter gradients)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
        self.optimizer.step()
        
        self.global_step += 1
        return loss.item()
    
    def train(self, num_epochs, log_interval=50, save_interval=500):
        """Training loop with automatic checkpointing"""
        
        print(f"\n{'='*60}")
        print(f"Starting Training - {self.person_id}")
        print(f"{'='*60}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{np.mean(epoch_losses):.4f}"
                })
                
                # Periodic logging
                if self.global_step % log_interval == 0:
                    avg_loss = np.mean(self.losses[-log_interval:])
                    print(f"\n[{self.person_id}] Step {self.global_step} | Loss: {avg_loss:.4f}")
                
                # Periodic checkpoint saving (for long training runs)
                if self.global_step % save_interval == 0:
                    self.ckpt_manager.save_checkpoint(
                        self.adapter,
                        self.optimizer,
                        epoch,
                        self.global_step,
                        np.mean(epoch_losses),
                        self.person_id
                    )
            
            # End of epoch checkpoint
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"\n✓ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}\n")
            
            self.ckpt_manager.save_checkpoint(
                self.adapter,
                self.optimizer,
                epoch + 1,
                self.global_step,
                avg_epoch_loss,
                self.person_id
            )
        
        print(f"\n{'='*60}")
        print(f"Training Complete - {self.person_id}")
        print(f"{'='*60}\n")


# ============================================================================
# DISTRIBUTED WORKFLOW HELPER
# ============================================================================

def merge_team_checkpoints(checkpoint_dir="checkpoints", person_ids=None):
    """
    Merge checkpoints from all team members
    
    Args:
        checkpoint_dir: directory containing checkpoints
        person_ids: list of person IDs to merge (if None, merges all *_latest.pt)
    """
    manager = CheckpointManager(checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    
    if person_ids is None:
        # Auto-detect all latest checkpoints
        checkpoint_paths = list(checkpoint_dir.glob("*_latest.pt"))
    else:
        # Use specific person IDs
        checkpoint_paths = [
            checkpoint_dir / f"adapter_{pid}_latest.pt"
            for pid in person_ids
        ]
    
    if len(checkpoint_paths) == 0:
        print("⚠ No checkpoints found to merge")
        return None
    
    # Merge and save
    merged_path = manager.merge_checkpoints(
        checkpoint_paths,
        output_name=f"merged_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    return merged_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # IMPORTANT: Each team member should set their own person_id!
    PERSON_ID = "Tristan"  # Change this for each person: "alice", "bob", "charlie", etc.
    
    CONFIG = {
        # Paths (each person can use their own dataset)
        'color_dir': 'dataset/images_color',
        'gray_dir': 'dataset/images_gray',
        'seg_dir': 'dataset/seg_maps',
        
        # Dataset
        'image_size': 512,
        
        # Model
        'base_channels': 64,
        'out_channels': 320,
        
        # Training
        'batch_size': 4,
        'learning_rate': 1e-5,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Checkpoints
        'checkpoint_dir': 'checkpoints',  # Shared checkpoint directory
        'log_interval': 50,
        'save_interval': 500,  # Save every 500 steps
        
        # Distributed training
        'person_id': PERSON_ID,
        'start_from_merged': None,  # Path to merged checkpoint to continue from
    }
    
    print("="*60)
    print("SEMANTIC ADAPTER - DISTRIBUTED TRAINING")
    print("="*60)
    print(f"Person ID: {CONFIG['person_id']}")
    print(f"Device: {CONFIG['device']}")
    print("="*60 + "\n")
    
    # ========================================================================
    # WORKFLOW SELECTION
    # ========================================================================
    
    print("Select workflow:")
    print("1. Train on your own dataset")
    print("2. Merge checkpoints from all team members")
    print("3. Continue training from merged checkpoint")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "2":
        # ====================================================================
        # MERGE WORKFLOW
        # ====================================================================
        print("\n" + "="*60)
        print("CHECKPOINT MERGING")
        print("="*60)
        
        # Option 1: Auto-merge all *_latest.pt files
        print("\nMerging all latest checkpoints...")
        merged_path = merge_team_checkpoints(CONFIG['checkpoint_dir'])
        
        # Option 2: Merge specific people (uncomment to use)
        # merged_path = merge_team_checkpoints(
        #     CONFIG['checkpoint_dir'],
        #     person_ids=['alice', 'bob', 'charlie']
        # )
        
        if merged_path:
            print(f"\n✓ SUCCESS!")
            print(f"Merged checkpoint saved at: {merged_path}")
            print(f"\nShare this file with your team to continue training!")
    
    elif choice in ["1", "3"]:
        # ====================================================================
        # TRAINING WORKFLOW
        # ====================================================================
        
        # Preprocess if needed
        if not Path(CONFIG['gray_dir']).exists():
            print("STEP: Preprocessing data...\n")
            convert_to_grayscale(CONFIG['color_dir'], CONFIG['gray_dir'])
            
            # Choose segmentation method:
            # Option 1: Simple YOLO (faster, only objects)
            # generate_segmentation_maps(CONFIG['color_dir'], CONFIG['seg_dir'])
            
            # Option 2: Hybrid (best results - objects + scenes)
            generate_hybrid_segmentation_maps(CONFIG['color_dir'], CONFIG['seg_dir'])
            
            print("\n✓ Preprocessing complete\n")
        
        # Load dataset
        print("STEP: Loading dataset...\n")
        dataset = SemanticColorDataset(
            gray_dir=CONFIG['gray_dir'],
            seg_dir=CONFIG['seg_dir'],
            color_dir=CONFIG['color_dir'],
            size=CONFIG['image_size']
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples\n")
        
        # Create or load model
        print("STEP: Creating Semantic Adapter...\n")
        adapter = SemanticAdapter(
            in_channels=1,
            base_channels=CONFIG['base_channels'],
            out_channels=CONFIG['out_channels']
        )
        
        num_params = sum(p.numel() for p in adapter.parameters())
        print(f"✓ Adapter created: {num_params:,} parameters\n")
        
        # Initialize trainer
        print("STEP: Initializing trainer...\n")
        trainer = SemanticAdapterTrainer(
            adapter=adapter,
            dataset=dataset,
            person_id=CONFIG['person_id'],
            lr=CONFIG['learning_rate'],
            batch_size=CONFIG['batch_size'],
            device=CONFIG['device']
        )
        
        # Load merged checkpoint if continuing (choice == "3")
        if choice == "3":
            if CONFIG['start_from_merged']:
                trainer.load_checkpoint(CONFIG['start_from_merged'])
            else:
                # Auto-find latest merged checkpoint
                merged_dir = Path(CONFIG['checkpoint_dir']) / "merged"
                if merged_dir.exists():
                    merged_files = sorted(list(merged_dir.glob("merged_adapter_*.pt")))
                    if merged_files:
                        latest_merged = merged_files[-1]
                        print(f"\n✓ Found merged checkpoint: {latest_merged}")
                        trainer.load_checkpoint(latest_merged)
        
        # Train
        print("\nStarting training...")
        trainer.train(
            num_epochs=CONFIG['num_epochs'],
            log_interval=CONFIG['log_interval'],
            save_interval=CONFIG['save_interval']
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nYour checkpoints are saved in: {CONFIG['checkpoint_dir']}")
        print(f"Latest checkpoint: adapter_{CONFIG['person_id']}_latest.pt")
        print("\nNext steps:")
        print("  1. Share your latest checkpoint with the team")
        print("  2. Collect checkpoints from all team members")
        print("  3. Run merge workflow (option 2) to combine everyone's work")
        print("  4. Continue training from merged checkpoint (option 3)")
        print("="*60)
    
    else:
        print("Invalid choice!")