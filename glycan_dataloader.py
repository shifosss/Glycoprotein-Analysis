"""
PyTorch DataLoader for Glycan-Protein Binding Data
Efficient GPU-based data loading for model training
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycanProteinDataset(Dataset):
    """
    PyTorch Dataset for glycan-protein binding data
    Returns pre-computed embeddings for efficient GPU training
    """

    def __init__(self,
                 embeddings: torch.Tensor,
                 targets: torch.Tensor,
                 device: Optional[str] = None):
        """
        Initialize dataset with pre-computed embeddings

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)
            targets: Tensor of binding strengths of shape (n_samples,)
            device: Device to store data on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move data to device
        self.embeddings = embeddings.to(self.device)
        self.targets = targets.to(self.device)

        assert len(self.embeddings) == len(self.targets), "Embeddings and targets must have same length"

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


class GlycanProteinDataLoader:
    """
    Efficient PyTorch DataLoader for glycan-protein binding data
    Handles data loading, embedding generation, and GPU operations
    """

    def __init__(self,
                 data_path: str,
                 embedder,  # GlycanProteinPairEmbedder instance
                 protein_col: str = 'protein',
                 sequence_col: str = 'target',
                 exclude_cols: Optional[List[str]] = None,
                 device: Optional[str] = None):
        """
        Initialize the DataLoader

        Args:
            data_path: Path to Excel file
            embedder: Pre-initialized GlycanProteinPairEmbedder
            protein_col: Column name for protein identifiers
            sequence_col: Column name for protein sequences
            exclude_cols: Additional columns to exclude
            device: Device for computations
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.protein_col = protein_col
        self.sequence_col = sequence_col
        self.exclude_cols = exclude_cols or []

        # Load and process data
        self.data = self._load_data(data_path)
        self.glycan_columns = self._get_glycan_columns()

        logger.info(f"Loaded data: {len(self.data)} samples, {len(self.glycan_columns)} glycans")
        logger.info(f"Using device: {self.device}")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        try:
            if str(data_path).endswith('.xlsx'):
                data = pd.read_excel(data_path, engine='openpyxl')
            elif str(data_path).endswith('.xls'):
                data = pd.read_excel(data_path, engine='xlrd')
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Remove rows with missing sequences
            data = data.dropna(subset=[self.sequence_col])

            # Fill missing binding values with median
            glycan_cols = [col for col in data.columns
                          if col not in {self.protein_col, self.sequence_col} | set(self.exclude_cols)]
            data[glycan_cols] = data[glycan_cols].fillna(data[glycan_cols].median())

            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _get_glycan_columns(self) -> List[str]:
        """Get list of glycan columns"""
        exclude_set = {self.protein_col, self.sequence_col} | set(self.exclude_cols)
        return [col for col in self.data.columns if col not in exclude_set]

    def create_pairs_dataset(self,
                           glycan_subset: Optional[List[str]] = None,
                           protein_subset: Optional[List[str]] = None,
                           max_pairs: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Create glycan-protein pairs and binding strengths

        Args:
            glycan_subset: Subset of glycans to use
            protein_subset: Subset of proteins to use
            max_pairs: Maximum number of pairs (for sampling)

        Returns:
            Tuple of (pairs, strengths)
        """
        # Filter data
        data = self.data.copy()
        if protein_subset:
            data = data[data[self.protein_col].isin(protein_subset)]

        # Get glycan columns to use
        glycans_to_use = glycan_subset if glycan_subset else self.glycan_columns

        pairs = []
        strengths = []

        # Create all pairs
        for _, row in data.iterrows():
            protein_sequence = row[self.sequence_col]

            for glycan in glycans_to_use:
                if glycan in data.columns:
                    binding_strength = row[glycan]
                    pairs.append((glycan, protein_sequence))
                    strengths.append(float(binding_strength))

        # Sample if max_pairs specified
        if max_pairs and len(pairs) > max_pairs:
            indices = np.random.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in indices]
            strengths = [strengths[i] for i in indices]
            logger.info(f"Sampled {max_pairs} pairs from {len(pairs)} total")

        return pairs, strengths

    def split_by_protein(self,
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        random_state: int = 42) -> Dict[str, List[str]]:
        """
        Split proteins for train/val/test (prevents data leakage)

        Args:
            test_size: Fraction for test
            val_size: Fraction for validation
            random_state: Random seed

        Returns:
            Dictionary with protein lists for each split
        """
        from sklearn.model_selection import train_test_split

        unique_proteins = self.data[self.protein_col].unique()

        # Split proteins
        train_proteins, test_proteins = train_test_split(
            unique_proteins, test_size=test_size, random_state=random_state
        )

        if val_size > 0:
            train_proteins, val_proteins = train_test_split(
                train_proteins, test_size=val_size/(1-test_size), random_state=random_state
            )
        else:
            val_proteins = []

        return {
            'train': train_proteins.tolist(),
            'val': val_proteins.tolist() if len(val_proteins) > 0 else [],
            'test': test_proteins.tolist()
        }

    def create_pytorch_dataloader(self,
                                protein_subset: Optional[List[str]] = None,
                                glycan_subset: Optional[List[str]] = None,
                                batch_size: int = 32,
                                shuffle: bool = True,
                                num_workers: int = 0,
                                normalize_targets: bool = True,
                                embedding_batch_size: int = 32,
                                max_pairs: Optional[int] = None) -> Tuple[DataLoader, Optional[torch.nn.Module]]:
        """
        Create PyTorch DataLoader with pre-computed embeddings

        Args:
            protein_subset: Proteins to include
            glycan_subset: Glycans to include
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            normalize_targets: Whether to normalize binding strengths
            embedding_batch_size: Batch size for embedding computation
            max_pairs: Maximum number of pairs to use

        Returns:
            Tuple of (DataLoader, target_scaler)
        """
        # Create pairs and strengths
        pairs, strengths = self.create_pairs_dataset(
            glycan_subset=glycan_subset,
            protein_subset=protein_subset,
            max_pairs=max_pairs
        )

        logger.info(f"Computing embeddings for {len(pairs)} pairs...")

        # Compute embeddings in batches
        embeddings = self.embedder.embed_pairs(
            pairs,
            batch_size=embedding_batch_size,
            return_numpy=False
        )

        # Convert to tensors
        strengths_tensor = torch.FloatTensor(strengths)

        # Normalize targets if requested
        target_scaler = None
        if normalize_targets:
            from sklearn.preprocessing import StandardScaler
            target_scaler = StandardScaler()
            strengths_normalized = target_scaler.fit_transform(
                strengths_tensor.numpy().reshape(-1, 1)
            ).flatten()
            strengths_tensor = torch.FloatTensor(strengths_normalized)

        # Create dataset
        dataset = GlycanProteinDataset(
            embeddings=embeddings,
            targets=strengths_tensor,
            device=self.device
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}")

        return dataloader, target_scaler

    def create_train_val_test_loaders(self,
                                    test_size: float = 0.2,
                                    val_size: float = 0.1,
                                    batch_size: int = 32,
                                    embedding_batch_size: int = 32,
                                    normalize_targets: bool = True,
                                    random_state: int = 42,
                                    max_pairs_per_split: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Create train/validation/test DataLoaders with protein-based splitting

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            batch_size: Batch size for training
            embedding_batch_size: Batch size for embedding computation
            normalize_targets: Whether to normalize targets
            random_state: Random seed
            max_pairs_per_split: Maximum pairs per split (for testing)

        Returns:
            Dictionary with DataLoaders and metadata
        """
        # Split proteins
        protein_splits = self.split_by_protein(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        dataloaders = {}
        target_scalers = {}

        # Create DataLoader for each split
        for split_name, proteins in protein_splits.items():
            if len(proteins) == 0:
                continue

            logger.info(f"Creating {split_name} DataLoader with {len(proteins)} proteins")

            dataloader, scaler = self.create_pytorch_dataloader(
                protein_subset=proteins,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                embedding_batch_size=embedding_batch_size,
                normalize_targets=normalize_targets,
                max_pairs=max_pairs_per_split
            )

            dataloaders[split_name] = dataloader
            target_scalers[split_name] = scaler

        # Store scaler for denormalization (use train scaler for consistency)
        dataloaders['target_scaler'] = target_scalers.get('train')

        # Log split sizes
        logger.info("DataLoader split sizes:")
        for split_name, dataloader in dataloaders.items():
            if isinstance(dataloader, DataLoader):
                logger.info(f"  {split_name}: {len(dataloader.dataset)} samples")

        return dataloaders


def create_glycan_dataloaders(data_path: str,
                            embedder,
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            batch_size: int = 32,
                            max_pairs: Optional[int] = None,
                            device: Optional[str] = None) -> Dict[str, DataLoader]:
    """
    Convenience function to create train/val/test DataLoaders

    Args:
        data_path: Path to Excel data file
        embedder: GlycanProteinPairEmbedder instance
        test_size: Test set fraction
        val_size: Validation set fraction
        batch_size: Batch size
        max_pairs: Maximum pairs per split (for testing)
        device: Computing device

    Returns:
        Dictionary with DataLoaders
    """
    loader = GlycanProteinDataLoader(
        data_path=data_path,
        embedder=embedder,
        device=device
    )

    return loader.create_train_val_test_loaders(
        test_size=test_size,
        val_size=val_size,
        batch_size=batch_size,
        max_pairs_per_split=max_pairs
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Glycan PyTorch DataLoader")

    # This would normally be imported
    # from Glycan_Protein_Integrated_Embedder import GlycanProteinPairEmbedder

    try:
        # Mock embedder for testing
        class MockEmbedder:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            def embed_pairs(self, pairs, batch_size=32, return_numpy=False):
                # Mock embedding computation
                n_pairs = len(pairs)
                embedding_dim = 128  # Mock dimension
                embeddings = torch.randn(n_pairs, embedding_dim)
                return embeddings.to(self.device) if not return_numpy else embeddings.numpy()

        # Create mock embedder
        embedder = MockEmbedder()

        # Test DataLoader creation
        data_path = "data/v12_glycan_binding.csv"

        loader = GlycanProteinDataLoader(
            data_path=data_path,
            embedder=embedder
        )

        # Create DataLoaders
        dataloaders = loader.create_train_val_test_loaders(
            batch_size=16,
            max_pairs_per_split=100  # Small for testing
        )

        # Test training loop
        print("\nTesting training loop:")
        train_loader = dataloaders['train']

        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: embeddings {embeddings.shape}, targets {targets.shape}")
            print(f"  Device: embeddings on {embeddings.device}, targets on {targets.device}")

            if batch_idx >= 2:  # Test first few batches
                break

        print(f"\nDataLoader test completed successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(dataloaders.get('val', []))}")
        print(f"Test batches: {len(dataloaders['test'])}")

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()