"""
Binding Strength Predictor
Integrates glycan-protein embedder with neural networks for supervised learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import custom modules
from embedder.Integrated_Embedder import GlycanProteinPairEmbedder
from network.binding_strength_networks import BindingStrengthNetworkFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BindingStrengthPredictor:
    """
    Main class for predicting protein-glycan binding strength using embeddings and neural networks
    """

    def __init__(self,
                 # Embedder parameters
                 protein_model: str = "650M",
                 protein_model_dir: str = "resources/esm-model-weights",
                 glycan_method: str = "lstm",
                 glycan_vocab_path: Optional[str] = None,
                 glycan_hidden_dims: Optional[List[int]] = None,
                 glycan_readout: str = "mean",
                 fusion_method: str = "concat",

                 # Network parameters
                 network_type: str = "mlp",
                 network_config: Optional[Dict] = None,

                 # Training parameters
                 device: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize the Binding Strength Predictor

        Args:
            protein_model: ESM2 model size ("650M" or "3B")
            protein_model_dir: Directory for protein model weights
            glycan_method: Glycan embedding method
            glycan_vocab_path: Path to glycan vocabulary file
            glycan_hidden_dims: Hidden dimensions for glycan embedder
            glycan_readout: Readout function for graph-based methods
            fusion_method: "concat" or "attention"
            network_type: Type of neural network ("mlp", "residual_mlp", "attention", "ensemble")
            network_config: Configuration for the neural network
            device: Device to use (None = auto-detect)
            random_seed: Random seed for reproducibility
        """
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize embedder
        logger.info("Initializing glycan-protein embedder...")
        self.embedder = GlycanProteinPairEmbedder(
            protein_model=protein_model,
            protein_model_dir=protein_model_dir,
            glycan_method=glycan_method,
            glycan_vocab_path=glycan_vocab_path,
            glycan_hidden_dims=glycan_hidden_dims,
            glycan_readout=glycan_readout,
            fusion_method=fusion_method,
            device=self.device
        )

        # Get embedding dimension
        self.embedding_dim = self.embedder.get_output_dim()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize network
        self.network_type = network_type
        self.network_config = network_config or BindingStrengthNetworkFactory.get_default_config(network_type)

        self.model = BindingStrengthNetworkFactory.create_network(
            network_type, self.embedding_dim, **self.network_config
        ).to(self.device)

        logger.info(f"Initialized {network_type} network with {sum(p.numel() for p in self.model.parameters())} parameters")

        # Initialize training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def prepare_data(self,
                     pairs: List[Tuple[str, str]],
                     strengths: List[float],
                     batch_size: int = 32,
                     val_split: float = 0.2,
                     test_split: float = 0.1,
                     normalize_targets: bool = True) -> Dict[str, DataLoader]:
        """
        Prepare data for training and evaluation

        Args:
            pairs: List of (glycan_iupac, protein_sequence) tuples
            strengths: List of binding strength values
            batch_size: Batch size for data loaders
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            normalize_targets: Whether to normalize target values

        Returns:
            Dictionary containing train, validation, and test data loaders
        """
        logger.info(f"Preparing data for {len(pairs)} samples...")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=True)

        # Convert to numpy arrays
        strengths = np.array(strengths)

        # Normalize targets if requested
        if normalize_targets:
            strengths = self.scaler.fit_transform(strengths.reshape(-1, 1)).flatten()
            logger.info("Normalized target values")

        # Convert to torch tensors
        X = torch.FloatTensor(embeddings)
        y = torch.FloatTensor(strengths)

        # Create dataset
        dataset = TensorDataset(X, y)

        # Split data
        n_samples = len(dataset)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_test - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

        # Create data loaders
        data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

        logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test}")

        return data_loaders

    def train(self,
              data_loaders: Dict[str, DataLoader],
              num_epochs: int = 100,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 10,
              min_delta: float = 1e-4,
              scheduler_config: Optional[Dict] = None) -> Dict:
        """
        Train the model

        Args:
            data_loaders: Dictionary with train/val/test data loaders
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for L2 regularization
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
            scheduler_config: Configuration for learning rate scheduler

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        # Initialize training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize scheduler if config provided
        if scheduler_config:
            scheduler_type = scheduler_config.pop('type', 'reduce_on_plateau')
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **scheduler_config
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=num_epochs, **scheduler_config
                )

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(data_loaders['train'])

            # Validation phase
            val_loss, val_metrics = self._eval_epoch(data_loaders['val'])

            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train R²={train_metrics['r2']:.4f}, Val R²={val_metrics['r2']:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        self.is_fitted = True

        return self.training_history

    def _train_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def _eval_epoch(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def _calculate_metrics(self, targets: List[float], predictions: List[float]) -> Dict:
        """Calculate evaluation metrics"""
        targets = np.array(targets)
        predictions = np.array(predictions)

        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions)
        }

    def predict(self,
                pairs: List[Tuple[str, str]],
                batch_size: int = 32,
                return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict binding strengths for new pairs

        Args:
            pairs: List of (glycan_iupac, protein_sequence) tuples
            batch_size: Batch size for processing
            return_numpy: Whether to return numpy array

        Returns:
            Predicted binding strengths
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        logger.info(f"Predicting binding strengths for {len(pairs)} pairs...")

        # Generate embeddings
        embeddings = self.embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=False)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].to(self.device)
                pred = self.model(batch).squeeze()
                predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)

        # Denormalize if scaler was used
        if hasattr(self.scaler, 'scale_'):
            predictions_np = predictions.cpu().numpy().reshape(-1, 1)
            predictions_np = self.scaler.inverse_transform(predictions_np).flatten()
            predictions = torch.FloatTensor(predictions_np)

        if return_numpy:
            return predictions.cpu().numpy()
        else:
            return predictions

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset"""
        _, metrics = self._eval_epoch(data_loader)
        return metrics

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # R² curves
        train_r2 = [m['r2'] for m in self.training_history['train_metrics']]
        val_r2 = [m['r2'] for m in self.training_history['val_metrics']]
        axes[0, 1].plot(train_r2, label='Train R²')
        axes[0, 1].plot(val_r2, label='Val R²')
        axes[0, 1].set_title('R² Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()

        # MAE curves
        train_mae = [m['mae'] for m in self.training_history['train_metrics']]
        val_mae = [m['mae'] for m in self.training_history['val_metrics']]
        axes[1, 0].plot(train_mae, label='Train MAE')
        axes[1, 0].plot(val_mae, label='Val MAE')
        axes[1, 0].set_title('MAE Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()

        # RMSE curves
        train_rmse = [m['rmse'] for m in self.training_history['train_metrics']]
        val_rmse = [m['rmse'] for m in self.training_history['val_metrics']]
        axes[1, 1].plot(train_rmse, label='Train RMSE')
        axes[1, 1].plot(val_rmse, label='Val RMSE')
        axes[1, 1].set_title('RMSE Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")

        plt.show()

    def save_model(self, path: str):
        """Save model and configuration"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'network_type': self.network_type,
            'network_config': self.network_config,
            'embedding_dim': self.embedding_dim,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }

        torch.save(save_dict, path)
        logger.info(f"Saved model to {path}")

    def load_model(self, path: str):
        """Load model and configuration"""
        save_dict = torch.load(path, map_location=self.device)

        # Recreate model with saved configuration
        self.network_type = save_dict['network_type']
        self.network_config = save_dict['network_config']
        self.embedding_dim = save_dict['embedding_dim']

        self.model = BindingStrengthNetworkFactory.create_network(
            self.network_type, self.embedding_dim, **self.network_config
        ).to(self.device)

        # Load state
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        self.training_history = save_dict['training_history']
        self.is_fitted = save_dict['is_fitted']

        logger.info(f"Loaded model from {path}")


def load_data_from_file(file_path: str) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    Load glycan-protein pairs and binding strengths from file
    Assumes file format with columns: glycan_iupac, protein_sequence, binding_strength

    Args:
        file_path: Path to data file (CSV, TSV, or JSON)

    Returns:
        Tuple of (pairs, strengths)
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        pairs = [(item['glycan_iupac'], item['protein_sequence']) for item in data]
        strengths = [item['binding_strength'] for item in data]

    elif file_path.suffix.lower() in ['.csv', '.tsv']:
        separator = ',' if file_path.suffix.lower() == '.csv' else '\t'
        df = pd.read_csv(file_path, separator=separator)
        pairs = list(zip(df['glycan_iupac'], df['protein_sequence']))
        strengths = df['binding_strength'].tolist()

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(pairs)} pairs from {file_path}")
    return pairs, strengths


if __name__ == "__main__":
    # Example usage
    vocab_path = "../embedder/GlycanEmbedder_Package/glycoword_vocab.pkl"

    # Example pairs and binding strengths (dummy data)
    pairs = [
        ("Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
         "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("Neu5Ac(a2-3)Gal(b1-4)Glc",
         "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP")
    ] * 50  # Repeat for more samples

    # Generate dummy binding strengths
    np.random.seed(42)
    strengths = np.random.normal(0.5, 0.2, len(pairs)).tolist()

    # Example 1: Simple MLP predictor
    print("Example 1: Training MLP predictor")
    predictor = BindingStrengthPredictor(
        protein_model="650M",
        protein_model_dir="../resources/esm-model-weights",
        glycan_method="lstm",
        glycan_vocab_path=vocab_path,
        fusion_method="concat",
        network_type="mlp",
        network_config={
            "hidden_dims": [512, 256, 128],
            "dropout": 0.3,
            "activation": "relu"
        }
    )

    # Prepare data
    data_loaders = predictor.prepare_data(
        pairs, strengths,
        batch_size=16,
        val_split=0.2,
        test_split=0.1
    )

    # Train model
    history = predictor.train(
        data_loaders,
        num_epochs=20,
        learning_rate=1e-3,
        patience=5,
        scheduler_config={'type': 'reduce_on_plateau', 'patience': 3, 'factor': 0.5}
    )

    # Evaluate on test set
    test_metrics = predictor.evaluate(data_loaders['test'])
    print(f"Test metrics: {test_metrics}")

    # Make predictions on new data
    new_pairs = pairs[:5]
    predictions = predictor.predict(new_pairs)
    print(f"Predictions for first 5 pairs: {predictions}")

    # Plot training history
    predictor.plot_training_history()

    # Save model
    predictor.save_model("binding_strength_model.pth")

    print("\nExample completed successfully!")