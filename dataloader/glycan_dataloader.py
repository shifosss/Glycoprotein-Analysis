"""
PyTorch DataLoader for Glycan-Protein Binding Data
Efficient GPU-based data loading with caching and memory management
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from pathlib import Path
import pickle
import hashlib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedGlycanProteinDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that loads embeddings from cache files
    Only loads data to GPU when needed, preventing OOM errors
    """

    def __init__(self,
                 cache_files: List[str],
                 targets: torch.Tensor,
                 device: Optional[str] = None,
                 gpu_batch_size: int = 256):
        """
        Initialize dataset with cached embedding files

        Args:
            cache_files: List of paths to cached embedding files
            targets: Tensor of binding strengths
            device: Device to load data to
            gpu_batch_size: Number of embeddings to keep in GPU memory at once
        """
        self.cache_files = cache_files
        self.targets = targets.cpu()  # Keep targets on CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_batch_size = gpu_batch_size

        # GPU memory cache - keeps only recent embeddings
        self.gpu_cache = {}
        self.gpu_cache_order = []

        assert len(self.cache_files) == len(self.targets), "Cache files and targets must have same length"

        logger.info(f"Initialized cached dataset: {len(self.cache_files)} samples")
        logger.info(f"GPU batch size: {gpu_batch_size} (max embeddings in GPU memory)")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        # Check if embedding is in GPU cache
        if idx in self.gpu_cache:
            embedding = self.gpu_cache[idx]
        else:
            # Load from disk
            embedding = np.load(self.cache_files[idx])
            embedding = torch.FloatTensor(embedding).to(self.device)

            # Add to GPU cache with memory management
            self._add_to_gpu_cache(idx, embedding)

        target = self.targets[idx].to(self.device)
        return embedding, target

    def _add_to_gpu_cache(self, idx: int, embedding: torch.Tensor):
        """Add embedding to GPU cache with LRU eviction"""
        # Remove oldest if cache is full
        if len(self.gpu_cache) >= self.gpu_batch_size:
            oldest_idx = self.gpu_cache_order.pop(0)
            del self.gpu_cache[oldest_idx]

        # Add new embedding
        self.gpu_cache[idx] = embedding
        self.gpu_cache_order.append(idx)


class GlycanProteinDataLoader:
    """
    Efficient PyTorch DataLoader with embedding caching and memory management
    """

    def __init__(self,
                 data_path: str = "data/v12_glycan_binding.csv",
                 embedder=None,
                 protein_col: str = 'target',
                 sequence_col: str = 'target',
                 exclude_cols: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 cache_dir: str = "embedding_cache"):
        """
        Initialize the DataLoader with caching

        Args:
            data_path: Path to CSV/Excel file
            embedder: Pre-initialized GlycanProteinPairEmbedder
            protein_col: Column name for protein identifiers/sequences
            sequence_col: Column name for protein sequences
            exclude_cols: Additional columns to exclude
            device: Device for computations
            cache_dir: Directory to store cached embeddings
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.protein_col = protein_col
        self.sequence_col = sequence_col
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Auto-exclude the 'protein' column
        self.exclude_cols = exclude_cols or []
        if 'protein' not in self.exclude_cols:
            self.exclude_cols.append('protein')

        # Check if embedder is provided
        if embedder is None:
            logger.warning("No embedder provided. You'll need to set it before creating DataLoaders.")
            return

        # Load and process data
        self.data = self._load_data(data_path)
        self.glycan_columns = self._get_glycan_columns()

        logger.info(f"Loaded data: {len(self.data)} samples, {len(self.glycan_columns)} glycans")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file"""
        try:
            if str(data_path).endswith('.csv'):
                data = pd.read_csv(data_path, engine='python')
            elif str(data_path).endswith('.xlsx'):
                data = pd.read_excel(data_path, engine='openpyxl')
            elif str(data_path).endswith('.xls'):
                data = pd.read_excel(data_path, engine='xlrd')
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            logger.info(f"Raw data shape: {data.shape}")

            # Remove rows with missing sequences
            data = data.dropna(subset=[self.sequence_col])

            # Get glycan columns (numeric columns only)
            potential_glycan_cols = [col for col in data.columns
                                   if col not in {self.protein_col, self.sequence_col} | set(self.exclude_cols)]

            # Filter to numeric columns only
            glycan_cols = []
            for col in potential_glycan_cols:
                try:
                    pd.to_numeric(data[col], errors='raise')
                    glycan_cols.append(col)
                except (ValueError, TypeError):
                    logger.warning(f"Excluding non-numeric column: {col}")

            logger.info(f"Identified {len(glycan_cols)} numeric glycan columns")

            # Fill missing binding values with median
            if glycan_cols:
                data[glycan_cols] = data[glycan_cols].fillna(data[glycan_cols].median())

            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _get_glycan_columns(self) -> List[str]:
        """Get list of glycan columns (numeric columns only)"""
        exclude_set = {self.protein_col, self.sequence_col} | set(self.exclude_cols)
        potential_glycan_cols = [col for col in self.data.columns if col not in exclude_set]

        glycan_cols = []
        for col in potential_glycan_cols:
            try:
                pd.to_numeric(self.data[col], errors='raise')
                glycan_cols.append(col)
            except (ValueError, TypeError):
                logger.debug(f"Excluding non-numeric column: {col}")

        return glycan_cols

    def _get_cache_key(self, glycan: str, protein: str) -> str:
        """Generate cache key for glycan-protein pair"""
        pair_str = f"{glycan}|{protein}"
        return hashlib.md5(pair_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"{cache_key}.npy"

    def _load_or_compute_embeddings(self, pairs: List[Tuple[str, str]],
                                  embedding_batch_size: int = 32) -> List[str]:
        """
        Load embeddings from cache or compute and cache them
        Returns list of cache file paths
        """
        cache_files = []
        pairs_to_compute = []
        indices_to_compute = []

        # Check which embeddings are already cached
        for i, (glycan, protein) in enumerate(pairs):
            cache_key = self._get_cache_key(glycan, protein)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                cache_files.append(str(cache_path))
            else:
                cache_files.append(None)  # Placeholder
                pairs_to_compute.append((glycan, protein))
                indices_to_compute.append(i)

        # Compute missing embeddings in batches
        if pairs_to_compute:
            logger.info(f"Computing {len(pairs_to_compute)} missing embeddings...")

            for i in range(0, len(pairs_to_compute), embedding_batch_size):
                batch_pairs = pairs_to_compute[i:i + embedding_batch_size]
                batch_indices = indices_to_compute[i:i + embedding_batch_size]

                # Compute embeddings for batch
                batch_embeddings = self.embedder.embed_pairs(
                    batch_pairs,
                    batch_size=embedding_batch_size,
                    return_numpy=True
                )

                # Save each embedding to cache
                for j, (glycan, protein) in enumerate(batch_pairs):
                    cache_key = self._get_cache_key(glycan, protein)
                    cache_path = self._get_cache_path(cache_key)

                    # Save embedding
                    np.save(cache_path, batch_embeddings[j])

                    # Update cache_files list
                    original_idx = batch_indices[j]
                    cache_files[original_idx] = str(cache_path)

                # ==========================================
                # 🗑️ EXPLICIT GPU MEMORY CLEANUP
                # ==========================================
                # Delete the batch embeddings from CPU memory
                del batch_embeddings

                # Force GPU memory cleanup after each embedding batch
                # This prevents GPU memory accumulation during embedding computation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("🗑️ GPU memory cleared after embedding batch")

                # Import gc for more thorough cleanup (optional but recommended)
                import gc
                gc.collect()

                logger.info(f"Cached embeddings batch {i//embedding_batch_size + 1}/{(len(pairs_to_compute)-1)//embedding_batch_size + 1}")

        return cache_files

    def create_pairs_dataset(self,
                           glycan_subset: Optional[List[str]] = None,
                           protein_subset: Optional[List[str]] = None,
                           max_pairs: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Create glycan-protein pairs and binding strengths"""
        if self.data is None:
            raise ValueError("Data not loaded. Embedder was None during initialization.")

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
        """Split proteins for train/val/test (prevents data leakage)"""
        from sklearn.model_selection import train_test_split

        unique_proteins = self.data[self.protein_col].unique()

        logger.info(f"Splitting {len(unique_proteins)} unique protein sequences:")
        logger.info(f"  Test size: {test_size:.1%}")
        logger.info(f"  Val size: {val_size:.1%}")
        logger.info(f"  Train size: {1-test_size-val_size:.1%}")

        # First split: separate test set
        train_val_proteins, test_proteins = train_test_split(
            unique_proteins, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation from remaining training data
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            train_proteins, val_proteins = train_test_split(
                train_val_proteins, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            train_proteins = train_val_proteins
            val_proteins = []

        splits = {
            'train': train_proteins.tolist(),
            'val': val_proteins.tolist() if len(val_proteins) > 0 else [],
            'test': test_proteins.tolist()
        }

        logger.info(f"Actual split:")
        for split_name, proteins in splits.items():
            logger.info(f"  {split_name} proteins: {len(proteins)}")

        return splits

    def create_pytorch_dataloader(self,
                                protein_subset: Optional[List[str]] = None,
                                glycan_subset: Optional[List[str]] = None,
                                batch_size: int = 32,
                                shuffle: bool = True,
                                num_workers: int = 0,
                                normalize_targets: bool = True,
                                embedding_batch_size: int = 32,
                                max_pairs: Optional[int] = None,
                                gpu_memory_limit: int = 512) -> Tuple[DataLoader, Optional[torch.nn.Module]]:
        """
        Create PyTorch DataLoader with cached embeddings and memory management

        Args:
            protein_subset: Proteins to include
            glycan_subset: Glycans to include
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            normalize_targets: Whether to normalize binding strengths
            embedding_batch_size: Batch size for embedding computation
            max_pairs: Maximum number of pairs to use
            gpu_memory_limit: Max embeddings to keep in GPU memory at once

        Returns:
            Tuple of (DataLoader, target_scaler)
        """
        # Create pairs and strengths
        pairs, strengths = self.create_pairs_dataset(
            glycan_subset=glycan_subset,
            protein_subset=protein_subset,
            max_pairs=max_pairs
        )

        logger.info(f"Processing {len(pairs)} pairs with caching...")

        # Load or compute embeddings (returns cache file paths)
        cache_files = self._load_or_compute_embeddings(pairs, embedding_batch_size)

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

        # Create cached dataset
        dataset = CachedGlycanProteinDataset(
            cache_files=cache_files,
            targets=strengths_tensor,
            device=self.device,
            gpu_batch_size=gpu_memory_limit
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False  # We manage GPU memory ourselves
        )

        logger.info(f"Created cached DataLoader: {len(dataset)} samples, batch_size={batch_size}")
        logger.info(f"GPU memory limit: {gpu_memory_limit} embeddings")

        return dataloader, target_scaler

    def create_train_val_test_loaders(self,
                                    test_size: float = 0.2,
                                    val_size: float = 0.1,
                                    batch_size: int = 32,
                                    embedding_batch_size: int = 32,
                                    normalize_targets: bool = True,
                                    random_state: int = 42,
                                    max_pairs_per_split: Optional[int] = None,
                                    gpu_memory_limit: int = 512) -> Dict[str, DataLoader]:
        """
        Create train/validation/test DataLoaders with caching and memory management

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            batch_size: Batch size for training
            embedding_batch_size: Batch size for embedding computation
            normalize_targets: Whether to normalize targets
            random_state: Random seed
            max_pairs_per_split: Maximum pairs per split
            gpu_memory_limit: Max embeddings in GPU memory per dataset

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
                max_pairs=max_pairs_per_split,
                gpu_memory_limit=gpu_memory_limit
            )

            dataloaders[split_name] = dataloader
            target_scalers[split_name] = scaler

        # Store scaler for denormalization
        dataloaders['target_scaler'] = target_scalers.get('train')

        # Log split sizes
        logger.info("DataLoader split sizes:")
        for split_name, dataloader in dataloaders.items():
            if isinstance(dataloader, DataLoader):
                logger.info(f"  {split_name}: {len(dataloader.dataset)} samples")

        return dataloaders

    def clear_cache(self):
        """Clear all cached embeddings"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Cleared embedding cache")

    def get_cache_info(self) -> Dict:
        """Get information about the cache"""
        if not self.cache_dir.exists():
            return {"cached_files": 0, "cache_size_mb": 0}

        cache_files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cached_files": len(cache_files),
            "cache_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


# Keep the original convenience function for backward compatibility
def create_glycan_dataloaders(data_path: str = "data/v12_glycan_binding.csv",
                            embedder=None,
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            batch_size: int = 32,
                            embedding_batch_size: int = 32,
                            max_pairs: Optional[int] = None,
                            device: Optional[str] = None,
                            cache_dir: str = "embedding_cache",
                            gpu_memory_limit: int = 512) -> Dict[str, DataLoader]:
    """
    Convenience function to create train/val/test DataLoaders with caching

    Args:
        data_path: Path to CSV/Excel data file
        embedder: GlycanProteinPairEmbedder instance
        test_size: Test set fraction
        val_size: Validation set fraction
        batch_size: Batch size
        embedding_batch_size: Batch size for embedding computation
        max_pairs: Maximum pairs per split
        device: Computing device
        cache_dir: Directory for caching embeddings
        gpu_memory_limit: Max embeddings in GPU memory (prevents OOM)

    Returns:
        Dictionary with DataLoaders
    """
    if embedder is None:
        raise ValueError("Embedder is required. Please provide a GlycanProteinPairEmbedder instance.")

    loader = GlycanProteinDataLoader(
        data_path=data_path,
        embedder=embedder,
        device=device,
        cache_dir=cache_dir
    )

    return loader.create_train_val_test_loaders(
        test_size=test_size,
        val_size=val_size,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        max_pairs_per_split=max_pairs,
        gpu_memory_limit=gpu_memory_limit
    )


if __name__ == "__main__":
    # Example usage with caching
    print("Testing Cached Glycan PyTorch DataLoader")

    try:
        # Mock embedder for testing
        class MockEmbedder:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            def embed_pairs(self, pairs, batch_size=32, return_numpy=False):
                n_pairs = len(pairs)
                embedding_dim = 128
                embeddings = np.random.randn(n_pairs, embedding_dim).astype(np.float32)
                return embeddings

        embedder = MockEmbedder()

        # Create DataLoader with caching
        loader = GlycanProteinDataLoader(
            data_path="../data/v12_glycan_binding.csv",
            embedder=embedder,
            cache_dir="test_cache"
        )

        # Check cache info
        cache_info = loader.get_cache_info()
        print(f"Cache info: {cache_info}")

        # Create DataLoaders
        dataloaders = loader.create_train_val_test_loaders(
            batch_size=16,
            max_pairs_per_split=100,
            gpu_memory_limit=64  # Small for testing
        )

        # Test training loop
        print("\nTesting cached training loop:")
        train_loader = dataloaders['train']

        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: embeddings {embeddings.shape}, targets {targets.shape}")
            print(f"  Device: embeddings on {embeddings.device}, targets on {targets.device}")

            if batch_idx >= 2:
                break

        # Show final cache info
        final_cache_info = loader.get_cache_info()
        print(f"\nFinal cache info: {final_cache_info}")

        print("Cached DataLoader test completed successfully!")

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()