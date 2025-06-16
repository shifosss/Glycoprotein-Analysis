"""
PyTorch Integration Example: Glycan-Protein Binding Prediction
Optimized for V100-32G GPU with caching and memory management
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm
import time

# Import our custom modules
from glycan_dataloader import GlycanProteinDataLoader, create_glycan_dataloaders
from Integrated_Embedder import GlycanProteinPairEmbedder
from binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchBindingPredictor:
    """
    PyTorch-based binding strength predictor using GPU-optimized DataLoaders with caching
    Optimized for V100-32G GPU
    """

    def __init__(self,
                 embedder: GlycanProteinPairEmbedder,
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,
                 device: Optional[str] = None):
        """
        Initialize the predictor

        Args:
            embedder: Pre-initialized embedder
            network_type: Type of neural network
            network_config: Network configuration
            device: Computing device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder

        # Get embedding dimension
        self.embedding_dim = embedder.get_output_dim()

        # Create network
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)
        self.model = BindingStrengthNetworkFactory.create_network(
            network_type, self.embedding_dim, **self.network_config
        ).to(self.device)

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

        logger.info(f"Initialized predictor on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self,
              dataloaders: Dict[str, torch.utils.data.DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        """
        Train the model using cached PyTorch DataLoaders

        Args:
            dataloaders: Dictionary with train/val DataLoaders
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Store target scaler
        self.target_scaler = dataloaders.get('target_scaler')

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

        train_loader = dataloaders['train']
        val_loader = dataloaders.get('val', None)

        # Log dataset sizes
        logger.info(f"Train batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_r2 = self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                val_loss, val_r2 = self._eval_epoch(val_loader)
            else:
                val_loss, val_r2 = train_loss, train_r2

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

            # Log progress with timing
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Time={epoch_time:.1f}s"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        return self.history

    def _train_epoch(self, dataloader) -> tuple:
        """Train for one epoch with enhanced logging"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Use tqdm for progress bar
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch_idx, (embeddings, targets) in enumerate(pbar):
            # Data is already on GPU from cached dataset
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(embeddings).squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)

        return avg_loss, r2

    def _eval_epoch(self, dataloader) -> tuple:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Validation", leave=False):
                outputs = self.model(embeddings).squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        r2 = self._calculate_r2(all_targets, all_preds)

        return avg_loss, r2

    def _calculate_r2(self, targets, predictions):
        """Calculate R² score"""
        from sklearn.metrics import r2_score
        return r2_score(targets, predictions)

    def evaluate(self, dataloader) -> Dict:
        """Evaluate model on a dataset"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for embeddings, targets in tqdm(dataloader, desc="Evaluating", leave=False):
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

    def predict_from_dataloader(self, dataloader) -> np.ndarray:
        """Make predictions using a DataLoader"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for embeddings, _ in tqdm(dataloader, desc="Predicting", leave=False):
                outputs = self.model(embeddings).squeeze()
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions)

        # Denormalize if scaler available
        if self.target_scaler:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions


def run_pytorch_pipeline():
    """Run complete PyTorch-based pipeline optimized for V100-32G"""
    print("Running PyTorch Glycan-Protein Binding Pipeline")
    print("Optimized for V100-32G GPU with Caching")
    print("=" * 60)

    # Configuration optimized for V100-32G (32GB GPU memory)
    data_path = "data/v12_glycan_binding.csv"
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # V100-32G optimized settings
    gpu_memory_limit = 128  # ~10GB for embeddings (plenty of room for model)
    batch_size = 16  # Large batches for V100
    embedding_batch_size = 16  # Fast embedding computation
    cache_dir = "v100_embedding_cache"

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = GlycanProteinPairEmbedder(
            protein_model="650M",  # Can use "3B" if you want larger model
            protein_model_dir="resources/esm-model-weights",
            glycan_method="lstm",  # Try "gcn" or "bert" for different methods
            glycan_vocab_path=vocab_path,
            fusion_method="concat",  # Try "attention" for more sophisticated fusion
            device=device
        )

        # Create DataLoaders with caching and V100 optimization
        logger.info("Creating cached DataLoaders...")
        start_time = time.time()

        dataloaders = create_glycan_dataloaders(
            data_path=data_path,
            embedder=embedder,

            # Data splits
            test_size=0.1,
            val_size=0.1,

            # V100-32G optimized settings
            batch_size=batch_size,
            gpu_memory_limit=gpu_memory_limit,  # 4096 embeddings ~10GB
            cache_dir=cache_dir,
            device=device,
            embedding_batch_size=embedding_batch_size,

            # For testing, limit pairs (remove for full dataset)
            max_pairs=300  # Remove this line for full dataset
        )

        setup_time = time.time() - start_time
        logger.info(f"DataLoader setup completed in {setup_time:.1f}s")

        # Show cache information
        if hasattr(dataloaders, 'get_cache_info'):
            cache_info = dataloaders.get_cache_info()
            logger.info(f"Cache info: {cache_info}")
        else:
            # Create temporary loader to check cache
            temp_loader = GlycanProteinDataLoader(
                data_path=data_path,
                embedder=embedder,
                cache_dir=cache_dir
            )
            cache_info = temp_loader.get_cache_info()
            logger.info(f"Cache: {cache_info['cached_files']} files, {cache_info['cache_size_mb']:.1f} MB")

        # Initialize predictor with V100-optimized network
        logger.info("Initializing predictor...")
        predictor = PyTorchBindingPredictor(
            embedder=embedder,
            network_type="mlp",
            network_config={
                "hidden_dims": [1024, 512, 256, 128],  # Larger network for V100
                "dropout": 0.3,
                "activation": "relu",
                "batch_norm": True  # Add batch norm for stability
            },
            device=device
        )

        # Train model
        logger.info("Training model...")
        history = predictor.train(
            dataloaders=dataloaders,
            num_epochs=50,  # More epochs for V100
            learning_rate=2e-3,  # Slightly higher LR for larger batches
            weight_decay=1e-4,
            patience=15  # More patience for larger dataset
        )

        # Evaluate on test set
        logger.info("Evaluating model...")
        test_metrics, test_preds, test_targets = predictor.evaluate(dataloaders['test'])

        # Print results
        print(f"\nTraining completed!")
        print(f"Best validation R²: {max(history['val_r2']):.4f}")
        print(f"Final validation R²: {history['val_r2'][-1]:.4f}")
        print(f"Test Results:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")

        # Show GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        # Plot results if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('V100-32G Training Results', fontsize=16)

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

            # Residuals plot
            residuals = test_preds - test_targets
            axes[1, 1].scatter(test_targets, residuals, alpha=0.6, s=20)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Actual Binding Strength')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Plot')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("v100_pytorch_pipeline_results.png", dpi=300, bbox_inches='tight')
            plt.show()

        except ImportError:
            logger.info("Matplotlib not available, skipping plots")

        print(f"\nV100-32G PyTorch pipeline completed successfully!")
        print(f"Total training time with caching: {setup_time:.1f}s setup + training time")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


def test_v100_dataloader():
    """Test cached DataLoader functionality on V100-32G"""
    print("Testing V100-32G Cached DataLoader")
    print("=" * 40)

    try:
        # Mock embedder for testing
        class MockEmbedder:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            def get_output_dim(self):
                return 1280  # ESM2-650M dimension

            def embed_pairs(self, pairs, batch_size=32, return_numpy=False):
                n_pairs = len(pairs)
                embedding_dim = 1280
                embeddings = np.random.randn(n_pairs, embedding_dim).astype(np.float32)
                return embeddings

        embedder = MockEmbedder()

        # Test with V100-32G optimized settings
        start_time = time.time()

        dataloaders = create_glycan_dataloaders(
            data_path="data/v12_glycan_binding.csv",
            embedder=embedder,
            batch_size=128,  # Large batch for V100
            gpu_memory_limit=2048,  # 2048 embeddings ~5GB
            cache_dir="test_v100_cache",
            max_pairs=500  # Small for testing
        )

        setup_time = time.time() - start_time

        # Test iteration performance
        print(f"DataLoader setup: {setup_time:.2f}s")
        print("Testing iteration performance:")

        train_loader = dataloaders['train']

        iteration_start = time.time()
        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: {embeddings.shape}, {targets.shape}")
            print(f"  Devices: {embeddings.device}, {targets.device}")
            print(f"  Memory: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB allocated")

            if batch_idx >= 3:
                break

        iteration_time = time.time() - iteration_start
        print(f"Iteration time: {iteration_time:.2f}s")

        # Show cache info
        temp_loader = GlycanProteinDataLoader(
            data_path="data/v12_glycan_binding.csv",
            embedder=embedder,
            cache_dir="test_v100_cache"
        )
        cache_info = temp_loader.get_cache_info()
        print(f"Cache: {cache_info['cached_files']} files, {cache_info['cache_size_mb']:.1f} MB")

        print("V100-32G DataLoader test completed successfully!")

    except Exception as e:
        print(f"DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()


def clear_cache():
    """Utility function to clear embedding cache"""
    cache_dir = "v100_embedding_cache"
    temp_loader = GlycanProteinDataLoader(
        data_path="data/v12_glycan_binding.csv",
        embedder=None,  # Won't be used for cache operations
        cache_dir=cache_dir
    )

    cache_info = temp_loader.get_cache_info()
    print(f"Current cache: {cache_info['cached_files']} files, {cache_info['cache_size_mb']:.1f} MB")

    response = input("Clear cache? (y/N): ")
    if response.lower() == 'y':
        temp_loader.clear_cache()
        print("Cache cleared!")
    else:
        print("Cache not cleared.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_v100_dataloader()
        elif sys.argv[1] == "clear_cache":
            clear_cache()
        else:
            print("Available commands: test, clear_cache")
    else:
        run_pytorch_pipeline()