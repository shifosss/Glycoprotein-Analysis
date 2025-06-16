"""
Integration Example: Glycan-Protein Binding Prediction Pipeline
Demonstrates how to use the dataloader with the binding strength predictor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import logging

# Import our custom modules
from glycan_binding_dataloader import GlycanBindingDataLoader
from binding_strength_predictor import BindingStrengthPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycanProteinPipeline:
    """
    Complete pipeline for glycan-protein binding prediction
    """

    def __init__(self,
                 data_path: str,
                 vocab_path: str = "GlycanEmbedder_Package/glycoword_vocab.pkl",
                 protein_model_dir: str = "resource/esm-model-weights"):
        """
        Initialize the pipeline

        Args:
            data_path: Path to the Excel data file
            vocab_path: Path to glycan vocabulary file
            protein_model_dir: Directory containing protein model weights
        """
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.protein_model_dir = protein_model_dir

        # Components
        self.data_loader = None
        self.predictor = None
        self.results = {}

    def load_and_analyze_data(self,
                              min_variance: float = 0.01,
                              max_pairs: Optional[int] = None):
        """
        Load and analyze the dataset

        Args:
            min_variance: Minimum variance for glycan filtering
            max_pairs: Maximum number of pairs to use (for testing)
        """
        logger.info("Loading and analyzing data...")

        # Load data
        self.data_loader = GlycanBindingDataLoader(data_path=self.data_path)

        # Basic statistics
        logger.info(f"Dataset contains:")
        logger.info(f"  - {len(self.data_loader.protein_data)} protein samples")
        logger.info(f"  - {len(self.data_loader.glycan_columns)} glycan structures")
        logger.info(f"  - {self.data_loader.protein_data['protein'].nunique()} unique proteins")

        # Filter low-variance glycans
        high_var_glycans = self.data_loader.filter_glycans_by_variance(min_variance)
        logger.info(f"Keeping {len(high_var_glycans)} glycans after variance filtering")

        # Get pairs and strengths
        pairs, strengths = self.data_loader.get_pairs_and_strengths(glycan_subset=high_var_glycans)

        # Limit pairs if specified (for faster testing)
        if max_pairs and len(pairs) > max_pairs:
            indices = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in indices]
            strengths = [strengths[i] for i in indices]
            logger.info(f"Sampled {max_pairs} pairs for testing")

        self.results['pairs'] = pairs
        self.results['strengths'] = strengths

        # Analyze data distribution
        self._analyze_data_distribution()

        return pairs, strengths

    def _analyze_data_distribution(self):
        """Analyze the distribution of binding strengths"""
        strengths = np.array(self.results['strengths'])

        logger.info(f"Binding strength statistics:")
        logger.info(f"  Mean: {strengths.mean():.3f}")
        logger.info(f"  Std: {strengths.std():.3f}")
        logger.info(f"  Min: {strengths.min():.3f}")
        logger.info(f"  Max: {strengths.max():.3f}")
        logger.info(f"  Positive values: {(strengths > 0).sum()} ({(strengths > 0).mean() * 100:.1f}%)")

    def split_data(self,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   split_by_protein: bool = True):
        """
        Split data into train/validation/test sets

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            split_by_protein: If True, split by protein to avoid leakage
        """
        logger.info("Splitting data...")

        if split_by_protein:
            # Use protein-based splitting to avoid data leakage
            splits = self.data_loader.split_by_protein(
                test_size=test_size,
                val_size=val_size
            )
        else:
            # Random splitting (may have data leakage)
            pairs = self.results['pairs']
            strengths = self.results['strengths']

            # Split into train and temp
            n_total = len(pairs)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)
            n_train = n_total - n_test - n_val

            indices = np.random.permutation(n_total)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            splits = {
                'train': ([pairs[i] for i in train_idx], [strengths[i] for i in train_idx]),
                'val': ([pairs[i] for i in val_idx], [strengths[i] for i in val_idx]),
                'test': ([pairs[i] for i in test_idx], [strengths[i] for i in test_idx])
            }

        self.results['splits'] = splits

        logger.info(f"Data split sizes:")
        logger.info(f"  Train: {len(splits['train'][0])} pairs")
        logger.info(f"  Val: {len(splits['val'][0])} pairs")
        logger.info(f"  Test: {len(splits['test'][0])} pairs")

        return splits

    def train_model(self,
                    network_type: str = "mlp",
                    protein_model: str = "650M",
                    glycan_method: str = "lstm",
                    fusion_method: str = "concat",
                    num_epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 1e-3):
        """
        Train the binding strength prediction model

        Args:
            network_type: Type of neural network
            protein_model: ESM2 model size
            glycan_method: Glycan embedding method
            fusion_method: How to combine embeddings
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info("Training model...")

        if 'splits' not in self.results:
            raise ValueError("Data not split. Call split_data() first.")

        # Initialize predictor
        self.predictor = BindingStrengthPredictor(
            protein_model=protein_model,
            protein_model_dir=self.protein_model_dir,
            glycan_method=glycan_method,
            glycan_vocab_path=self.vocab_path,
            fusion_method=fusion_method,
            network_type=network_type
        )

        # Prepare data for training
        splits = self.results['splits']

        # Note: The predictor expects the data in a different format
        # We need to prepare it properly
        train_pairs, train_strengths = splits['train']
        val_pairs, val_strengths = splits['val']
        test_pairs, test_strengths = splits['test']

        # Create combined training data
        all_train_pairs = train_pairs + val_pairs
        all_train_strengths = train_strengths + val_strengths

        # Prepare data loaders using the predictor's method
        data_loaders = self.predictor.prepare_data(
            pairs=all_train_pairs,
            strengths=all_train_strengths,
            batch_size=batch_size,
            val_split=len(val_pairs) / len(all_train_pairs),  # Keep same validation ratio
            test_split=0.0,  # We'll use our separate test set
            normalize_targets=True
        )

        # Train the model
        history = self.predictor.train(
            data_loaders=data_loaders,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=10,
            scheduler_config={'type': 'reduce_on_plateau', 'patience': 5, 'factor': 0.5}
        )

        self.results['training_history'] = history

        return history

    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")

        if self.predictor is None:
            raise ValueError("Model not trained. Call train_model() first.")

        splits = self.results['splits']
        test_pairs, test_strengths = splits['test']

        # Make predictions
        predictions = self.predictor.predict(test_pairs, return_numpy=True)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Denormalize if needed (the predictor handles this internally)
        test_strengths = np.array(test_strengths)

        metrics = {
            'mse': mean_squared_error(test_strengths, predictions),
            'mae': mean_absolute_error(test_strengths, predictions),
            'rmse': np.sqrt(mean_squared_error(test_strengths, predictions)),
            'r2': r2_score(test_strengths, predictions)
        }

        self.results['test_metrics'] = metrics
        self.results['test_predictions'] = predictions
        self.results['test_actuals'] = test_strengths

        logger.info(f"Test Results:")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")

        return metrics

    def plot_results(self, save_path: Optional[str] = None):
        """Plot training and evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training history
        if 'training_history' in self.results:
            history = self.results['training_history']

            axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # R² scores
            train_r2 = [m['r2'] for m in history['train_metrics']]
            val_r2 = [m['r2'] for m in history['val_metrics']]
            axes[0, 1].plot(train_r2, label='Train R²', alpha=0.7)
            axes[0, 1].plot(val_r2, label='Val R²', alpha=0.7)
            axes[0, 1].set_title('R² Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 2: Predictions vs Actuals
        if 'test_predictions' in self.results:
            actuals = self.results['test_actuals']
            predictions = self.results['test_predictions']

            axes[1, 0].scatter(actuals, predictions, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

            axes[1, 0].set_xlabel('Actual Binding Strength')
            axes[1, 0].set_ylabel('Predicted Binding Strength')
            axes[1, 0].set_title('Predictions vs Actuals')
            axes[1, 0].grid(True, alpha=0.3)

            # Add R² to plot
            r2 = self.results['test_metrics']['r2']
            axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}',
                            transform=axes[1, 0].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 3: Residuals
        if 'test_predictions' in self.results:
            residuals = predictions - actuals
            axes[1, 1].scatter(predictions, residuals, alpha=0.6, s=20)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Predicted Binding Strength')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residual Plot')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plots to {save_path}")

        plt.show()

    def analyze_feature_importance(self, n_top: int = 20):
        """
        Analyze which glycans are most predictive
        Note: This is a simplified analysis - for proper feature importance,
        you'd need to implement gradient-based methods
        """
        logger.info("Analyzing feature importance...")

        # Get glycan statistics from the data
        glycan_stats = self.data_loader.get_glycan_statistics()

        # Simple correlation-based importance
        pairs = self.results['pairs']
        strengths = self.results['strengths']

        glycan_importance = {}

        for glycan in self.data_loader.glycan_columns:
            glycan_pairs = [(p, s) for p, s in zip(pairs, strengths) if p[0] == glycan]
            if glycan_pairs:
                glycan_strengths = [s for _, s in glycan_pairs]
                # Use variance as a proxy for importance
                importance = np.var(glycan_strengths)
                glycan_importance[glycan] = importance

        # Sort by importance
        sorted_glycans = sorted(glycan_importance.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Top {n_top} most variable glycans:")
        for i, (glycan, importance) in enumerate(sorted_glycans[:n_top]):
            logger.info(f"  {i + 1:2d}. {glycan}: {importance:.4f}")

        return sorted_glycans[:n_top]

    def save_results(self, save_path: str):
        """Save all results to file"""
        import pickle

        # Create a summary dictionary
        summary = {
            'data_info': {
                'n_proteins': self.data_loader.protein_data['protein'].nunique(),
                'n_glycans': len(self.data_loader.glycan_columns),
                'n_total_pairs': len(self.results.get('pairs', [])),
            },
            'model_config': {
                'network_type': getattr(self.predictor, 'network_type', None),
                'fusion_method': getattr(self.predictor, 'fusion_method', None),
            },
            'results': self.results
        }

        with open(save_path, 'wb') as f:
            pickle.dump(summary, f)

        logger.info(f"Saved results to {save_path}")


def run_example_pipeline():
    """Run an example of the complete pipeline"""
    print("Running Glycan-Protein Binding Prediction Pipeline")
    print("=" * 60)

    # Configuration
    data_path = "glycan_binding_example v12.xlsx"
    vocab_path = "GlycanEmbedder_Package/glycoword_vocab.pkl"

    try:
        # Initialize pipeline
        pipeline = GlycanProteinPipeline(
            data_path=data_path,
            vocab_path=vocab_path
        )

        # Load and analyze data (limit pairs for faster testing)
        pairs, strengths = pipeline.load_and_analyze_data(
            min_variance=0.1,  # Higher threshold for faster processing
            max_pairs=1000  # Limit for testing
        )

        # Split data
        splits = pipeline.split_data(split_by_protein=True)

        # Train model
        history = pipeline.train_model(
            network_type="mlp",
            glycan_method="lstm",
            fusion_method="concat",
            num_epochs=20,  # Reduced for faster testing
            batch_size=32
        )

        # Evaluate model
        metrics = pipeline.evaluate_model()

        # Plot results
        pipeline.plot_results("pipeline_results.png")

        # Analyze feature importance
        top_glycans = pipeline.analyze_feature_importance(n_top=10)

        # Save results
        pipeline.save_results("pipeline_results.pkl")

        print("\nPipeline completed successfully!")
        print(f"Final R² score: {metrics['r2']:.4f}")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_example_pipeline()