"""
GPU-Optimized PyTorch Training Example: Glycan-Protein Binding Prediction
Implements efficient GPU utilization with multi-worker data loading
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Import enhanced modules
from gpu_optim_dataloader import EnhancedGlycanProteinDataLoader, create_enhanced_glycan_dataloaders
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPyTorchBindingPredictor:
    """
    GPU-optimized PyTorch-based binding strength predictor
    """

    def __init__(self,
                 embedder: Optional[GlycanProteinPairEmbedder] = None,
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None,
                 use_precomputed: bool = True,
                 use_clustering: bool = True):
        """
        Initialize the predictor

        Args:
            embedder: Pre-initialized embedder (optional when using precomputed)
            network_type: Type of neural network
            network_config: Network configuration
            device: Computing device
            use_precomputed: Whether to use precomputed embeddings
            use_clustering: Whether to use clustering-based splits
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering

        # Get embedding dimension (will be determined from data if using precomputed)
        if embedder is not None:
            self.embedding_dim = embedder.get_output_dim()
        else:
            self.embedding_dim = None

        # Network configuration
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)

        # Model will be initialized after determining embedding dimension
        self.model = None

        # Training components
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.target_scaler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }

        logger.info(f"Initialized GPU-optimized predictor on {self.device}")

    def _initialize_model(self, embedding_dim: int):
        """Initialize model once embedding dimension is known"""
        if self.model is None:
            self.embedding_dim = embedding_dim
            self.model = BindingStrengthNetworkFactory.create_network(
                self.network_type, self.embedding_dim, **self.network_config
            ).to(self.device)

            # Ensure float32 precision
            self.model = self.model.float()

            logger.info(f"Initialized {self.network_type} network")
            logger.info(f"Embedding dim: {self.embedding_dim}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self,
              dataloaders: Dict[str, torch.utils.data.DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10,
              log_interval: int = 10) -> Dict:
        """
        Train the model with GPU optimization

        Args:
            dataloaders: Dictionary with train/val DataLoaders
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            log_interval: Interval for logging GPU stats

        Returns:
            Training history
        """
        logger.info(f"Starting GPU-optimized training for {num_epochs} epochs")

        # Store target scaler
        self.target_scaler = dataloaders.get('target_scaler')

        # Initialize model if not already done
        train_loader = dataloaders['train']

        # Get a sample batch to determine embedding dimension
        sample_batch = next(iter(train_loader))
        embedding_dim = sample_batch[0].shape[1]
        self._initialize_model(embedding_dim)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience // 2, factor=0.5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        val_loader = dataloaders.get('val', None)

        # Log dataset sizes
        logger.info(f"Train batches: {len(train_loader)}, Train samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}, Val samples: {len(val_loader.dataset)}")

        # GPU warm-up
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_r2, train_time = self._train_epoch(train_loader, epoch, log_interval)

            # Validation phase
            if val_loader:
                val_loss, val_r2, val_time = self._eval_epoch(val_loader)
            else:
                val_loss, val_r2, val_time = train_loss, train_r2, 0

            epoch_time = time.time() - epoch_start

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, "
                    f"Time={epoch_time:.1f}s (train={train_time:.1f}s, val={val_time:.1f}s)"
                )

                # Log GPU stats
                if self.device.startswith("cuda") and epoch % log_interval == 0:
                    self._log_gpu_stats()

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        # Final GPU cleanup
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return self.history

    def _train_epoch(self, dataloader, epoch, log_interval) -> tuple:
        """Train for one epoch with GPU optimization"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        start_time = time.time()

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)

        for batch_idx, (embeddings, targets) in enumerate(pbar):
            # Transfer to GPU with non-blocking
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Forward pass
            outputs = self.model(embeddings).squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate metrics (detach to prevent memory accumulation)
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Periodic GPU memory cleanup
            if batch_idx % 100 == 0 and self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)
        train_time = time.time() - start_time

        return avg_loss, r2, train_time

    def _eval_epoch(self, dataloader) -> tuple:
        """Evaluate for one epoch with GPU optimization"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        start_time = time.time()

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Validation", leave=False):
                # Transfer to GPU with non-blocking
                embeddings = embeddings.to(self.device, non_blocking=True).float()
                targets = targets.to(self.device, non_blocking=True).float()

                outputs = self.model(embeddings).squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)
        eval_time = time.time() - start_time

        return avg_loss, r2, eval_time

    def _calculate_r2(self, targets, predictions):
        """Calculate R² score"""
        from sklearn.metrics import r2_score
        return r2_score(targets, predictions)

    def _log_gpu_stats(self):
        """Log GPU memory statistics"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 ** 3
            max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3

            logger.info(f"GPU Memory - Current: {current_memory:.2f}GB, "
                        f"Max: {max_memory:.2f}GB, Reserved: {reserved_memory:.2f}GB")

    def evaluate(self, dataloader) -> Dict:
        """Evaluate model on a dataset"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        if self.model is None:
            raise ValueError("Model not initialized. Train the model first.")

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Evaluating", leave=False):
                embeddings = embeddings.to(self.device, non_blocking=True).float()
                targets = targets.to(self.device, non_blocking=True).float()

                outputs = self.model(embeddings).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Denormalize if scaler available
        if self.target_scaler:
            all_preds = self.target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
            all_targets = self.target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

        metrics = {
            'mse': mean_squared_error(all_targets, all_preds),
            'mae': mean_absolute_error(all_targets, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'r2': r2_score(all_targets, all_preds)
        }

        return metrics, all_preds, all_targets


def run_enhanced_pytorch_pipeline():
    """Run GPU-optimized PyTorch pipeline"""
    print("Running GPU-Optimized PyTorch Glycan-Protein Binding Pipeline")
    print("=" * 80)

    # Configuration
    data_path = "data/v12_glycan_binding.csv"
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enhanced settings for GPU optimization
    batch_size = 64  # Larger batch size for GPU
    num_workers = 4  # Multi-worker data loading
    cache_size_gb = 4.0  # In-memory cache size

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

    try:
        # Step 1: Create GPU-optimized DataLoaders
        logger.info("Creating GPU-optimized DataLoaders...")
        start_time = time.time()

        dataloaders = create_enhanced_glycan_dataloaders(
            data_path=data_path,
            # Data splits
            test_size=0.15,
            val_size=0.15,
            # Enhanced features
            use_precomputed=True,
            use_clustering=True,
            n_clusters=10,
            # GPU optimization
            batch_size=batch_size,
            num_workers=num_workers,
            cache_size_gb=cache_size_gb,
            preload_to_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            # Embedding settings
            protein_model="650M",
            glycan_method="lstm",
            glycan_vocab_path=vocab_path,
            # DataLoader settings
            cache_dir="enhanced_embeddings",
            device=device,
            # For testing, limit pairs (remove for full dataset)
            max_pairs=5000  # Remove this line for full dataset
        )

        setup_time = time.time() - start_time
        logger.info(f"DataLoader setup completed in {setup_time:.1f}s")

        # Step 2: Initialize GPU-optimized predictor
        logger.info("Initializing GPU-optimized predictor...")
        predictor = EnhancedPyTorchBindingPredictor(
            network_type="mlp",
            network_config={
                "hidden_dims": [1024, 512, 256, 128],
                "dropout": 0.3,
                "activation": "relu",
                "batch_norm": True
            },
            device=device,
            use_precomputed=True,
            use_clustering=True
        )

        # Step 3: Train model with GPU optimization
        logger.info("Training with GPU optimization...")
        training_start = time.time()

        history = predictor.train(
            dataloaders=dataloaders,
            num_epochs=50,
            learning_rate=2e-3,
            weight_decay=1e-4,
            patience=15,
            log_interval=10  # Log GPU stats every 10 epochs
        )

        training_time = time.time() - training_start

        # Step 4: Evaluate on test set
        logger.info("Evaluating model...")
        test_metrics, test_preds, test_targets = predictor.evaluate(dataloaders['test'])

        # Print results
        print(f"\nGPU-Optimized Training Results:")
        print(f"Setup time: {setup_time:.1f}s")
        print(f"Training time: {training_time:.1f}s")
        print(f"Total time: {setup_time + training_time:.1f}s")
        print(f"\nModel Performance:")
        print(f"Best validation R²: {max(history['val_r2']):.4f}")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")
        print(f"\nTest Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")

        # Show final GPU memory usage
        if torch.cuda.is_available():
            predictor._log_gpu_stats()

        # Step 5: Create visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('GPU-Optimized Training Results', fontsize=16)

            # Training curves
            axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # R² curves
            axes[0, 1].plot(history['train_r2'], label='Train R²', alpha=0.7)
            axes[0, 1].plot(history['val_r2'], label='Val R²', alpha=0.7)
            axes[0, 1].set_title('R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Predictions vs targets
            axes[1, 0].scatter(test_targets, test_preds, alpha=0.6, s=20)
            min_val = min(test_targets.min(), test_preds.min())
            max_val = max(test_targets.max(), test_preds.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 0].set_xlabel('Actual Binding Strength')
            axes[1, 0].set_ylabel('Predicted Binding Strength')
            axes[1, 0].set_title(f'Predictions vs Actuals (R² = {test_metrics["r2"]:.3f})')
            axes[1, 0].grid(True, alpha=0.3)

            # Optimization info
            optimization_text = f"""
GPU Optimizations Applied:
• Multi-worker data loading ({num_workers} workers)
• In-memory caching ({cache_size_gb} GB)
• Larger batch size ({batch_size})
• Non-blocking GPU transfers
• Persistent workers
• Prefetch factor: 2
• Float32 precision
• Gradient accumulation: No

Performance:
• Samples/sec: {len(dataloaders['train'].dataset) * len(history['train_loss']) / training_time:.1f}
• GPU Utilization: Check nvidia-smi
            """

            axes[1, 1].text(0.1, 0.5, optimization_text, transform=axes[1, 1].transAxes,
                            fontsize=9, verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            axes[1, 1].set_title('GPU Optimization Summary')
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig("gpu_optimized_training_results.png", dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            logger.info("Matplotlib not available, skipping plots")

        print(f"\nGPU-optimized pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in GPU-optimized pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_pytorch_pipeline()