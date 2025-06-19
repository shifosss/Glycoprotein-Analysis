"""
Enhanced PyTorch DataLoader for Glycan-Protein Binding Data (GPU-Optimized Version)
Implements in-memory caching and GPU-aware data loading
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
from collections import OrderedDict

from embedding_preprocessor import EmbeddingPreprocessor
from clustering_splitter import ProteinClusteringSplitter, create_clustered_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUOptimizedGlycanProteinDataset(Dataset):
    """
    GPU-optimized dataset with in-memory caching and iterative GPU loading
    """

    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 targets: torch.Tensor,
                 protein_cache_mapping: Dict[str, str],
                 glycan_cache_mapping: Dict[str, str],
                 fusion_method: str = "concat",
                 device: Optional[str] = None,
                 cache_size_gb: float = 4.0,
                 preload_to_memory: bool = True):
        """
        Initialize GPU-optimized dataset

        Args:
            pairs: List of (glycan_iupac, protein_sequence) tuples
            targets: Tensor of binding strengths
            protein_cache_mapping: Mapping from protein sequences to cache files
            glycan_cache_mapping: Mapping from glycan IUPACs to cache files
            fusion_method: "concat" or "attention"
            device: Device to load data to
            cache_size_gb: Maximum cache size in GB for in-memory storage
            preload_to_memory: Whether to preload embeddings to memory
        """
        self.pairs = pairs
        self.targets = targets.float()  # Ensure float32
        self.protein_cache_mapping = protein_cache_mapping
        self.glycan_cache_mapping = glycan_cache_mapping
        self.fusion_method = fusion_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_size_gb = cache_size_gb
        self.preload_to_memory = preload_to_memory

        assert len(self.pairs) == len(self.targets), "Pairs and targets must have same length"

        # In-memory cache with LRU eviction
        self.memory_cache = OrderedDict()
        self.cache_size_bytes = cache_size_gb * 1024 * 1024 * 1024
        self.current_cache_size = 0

        # GPU cache for frequently accessed embeddings
        self.gpu_cache = {}
        self.gpu_cache_hits = {}
        self.max_gpu_cache_items = 1000  # Adjust based on GPU memory

        # Preload if requested
        if preload_to_memory:
            self._preload_embeddings()

        logger.info(f"Initialized GPU-optimized dataset: {len(self.pairs)} samples")
        logger.info(f"Memory cache size: {cache_size_gb} GB, GPU cache: {self.max_gpu_cache_items} items")

    def _preload_embeddings(self):
        """Preload unique embeddings to memory cache"""
        unique_proteins = set(seq for _, seq in self.pairs)
        unique_glycans = set(glycan for glycan, _ in self.pairs)

        logger.info(f"Preloading {len(unique_proteins)} proteins and {len(unique_glycans)} glycans...")

        # Preload proteins
        for protein in unique_proteins:
            if protein in self.protein_cache_mapping:
                self._load_to_memory_cache(
                    self.protein_cache_mapping[protein],
                    f"protein_{protein[:50]}"
                )

        # Preload glycans
        for glycan in unique_glycans:
            if glycan in self.glycan_cache_mapping:
                self._load_to_memory_cache(
                    self.glycan_cache_mapping[glycan],
                    f"glycan_{glycan}"
                )

        logger.info(f"Preloaded {len(self.memory_cache)} embeddings, "
                   f"cache size: {self.current_cache_size / 1024 / 1024:.1f} MB")

    def _load_to_memory_cache(self, path: str, key: str) -> torch.Tensor:
        """Load embedding to memory cache with LRU eviction"""
        if key in self.memory_cache:
            # Move to end (most recently used)
            self.memory_cache.move_to_end(key)
            return self.memory_cache[key]

        # Load from disk
        embedding = torch.from_numpy(np.load(path)).float()  # float32
        embedding_size = embedding.element_size() * embedding.nelement()

        # Evict old items if needed
        while self.current_cache_size + embedding_size > self.cache_size_bytes and self.memory_cache:
            oldest_key = next(iter(self.memory_cache))
            oldest_embedding = self.memory_cache.pop(oldest_key)
            self.current_cache_size -= oldest_embedding.element_size() * oldest_embedding.nelement()

        # Add to cache
        self.memory_cache[key] = embedding
        self.current_cache_size += embedding_size

        return embedding

    def _get_embedding(self, path: str, key: str) -> torch.Tensor:
        """Get embedding with hierarchical caching"""
        # Check GPU cache first
        if key in self.gpu_cache:
            self.gpu_cache_hits[key] = self.gpu_cache_hits.get(key, 0) + 1
            return self.gpu_cache[key]

        # Check memory cache
        if key in self.memory_cache:
            embedding = self.memory_cache[key]
        else:
            # Load from disk to memory cache
            embedding = self._load_to_memory_cache(path, key)

        # Promote frequently accessed embeddings to GPU
        hits = self.gpu_cache_hits.get(key, 0) + 1
        self.gpu_cache_hits[key] = hits

        if hits >= 3 and len(self.gpu_cache) < self.max_gpu_cache_items:
            # Move to GPU cache
            self.gpu_cache[key] = embedding.to(self.device, non_blocking=True)

        return embedding

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get item with optimized caching"""
        glycan_iupac, protein_sequence = self.pairs[idx]

        # Get protein embedding
        protein_path = self.protein_cache_mapping[protein_sequence]
        protein_key = f"protein_{protein_sequence[:50]}"
        protein_emb = self._get_embedding(protein_path, protein_key)

        # Get glycan embedding
        glycan_path = self.glycan_cache_mapping[glycan_iupac]
        glycan_key = f"glycan_{glycan_iupac}"
        glycan_emb = self._get_embedding(glycan_path, glycan_key)

        # Normalize (on CPU to avoid repeated GPU operations)
        if protein_emb.device.type == 'cpu':
            protein_emb = nn.functional.normalize(protein_emb, p=2, dim=0)
            glycan_emb = nn.functional.normalize(glycan_emb, p=2, dim=0)

        # Combine embeddings
        if self.fusion_method == "concat":
            combined_emb = torch.cat([glycan_emb, protein_emb], dim=0)
        else:
            combined_emb = torch.cat([glycan_emb, protein_emb], dim=0)

        target = self.targets[idx]

        # Return CPU tensors - DataLoader will handle GPU transfer with pin_memory
        if combined_emb.device.type != 'cpu':
            combined_emb = combined_emb.cpu()

        return combined_emb, target

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        self.gpu_cache.clear()
        self.gpu_cache_hits.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EnhancedGlycanProteinDataLoader:
    """
    Enhanced DataLoader with GPU optimization
    """

    def __init__(self,
                 data_path: str = "data/v12_glycan_binding.csv",
                 embedder=None,
                 protein_col: str = 'target',
                 sequence_col: str = 'target',
                 exclude_cols: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 cache_dir: str = "preprocessed_embeddings",
                 use_precomputed: bool = True,
                 use_clustering: bool = True,
                 clustering_params: Optional[Dict] = None):
        """
        Initialize the enhanced DataLoader (unchanged interface)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder
        self.data_path = data_path
        self.protein_col = protein_col
        self.sequence_col = sequence_col
        self.cache_dir = Path(cache_dir)
        self.use_precomputed = use_precomputed
        self.use_clustering = use_clustering

        # Auto-exclude the 'protein' column
        self.exclude_cols = exclude_cols or []
        if 'protein' not in self.exclude_cols:
            self.exclude_cols.append('protein')

        # Clustering parameters
        self.clustering_params = clustering_params or {
            'n_clusters': 10,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'use_pca': True
        }

        # Initialize components
        self.preprocessor = None
        self.splitter = None
        self.protein_cache_mapping = None
        self.glycan_cache_mapping = None

        # Check if embedder is provided for backward compatibility
        if embedder is None and not use_precomputed:
            logger.warning("No embedder provided and precomputed embeddings disabled. "
                          "You'll need to set embedder or enable precomputed embeddings.")
            return

        # Load and process data
        self.data = self._load_data(data_path)
        self.glycan_columns = self._get_glycan_columns()

        logger.info(f"Loaded data: {len(self.data)} samples, {len(self.glycan_columns)} glycans")
        logger.info(f"Using device: {self.device}")

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

    def setup_precomputed_embeddings(self,
                                   protein_model: str = "650M",
                                   glycan_method: str = "lstm",
                                   glycan_vocab_path: Optional[str] = None,
                                   glycan_hidden_dims: Optional[List[int]] = None,
                                   glycan_readout: str = "mean",
                                   force_recompute: bool = False,
                                   **kwargs):
        """Setup precomputed embeddings (unchanged)"""
        if not self.use_precomputed:
            logger.info("Precomputed embeddings disabled, skipping setup")
            return

        logger.info("Setting up precomputed embeddings...")

        # Check if mappings already exist
        mapping_file = self.cache_dir / "cache_mappings.pkl"
        if mapping_file.exists() and not force_recompute:
            logger.info("Loading existing embedding mappings...")
            with open(mapping_file, 'rb') as f:
                mappings = pickle.load(f)
                self.protein_cache_mapping = mappings['protein_mapping']
                self.glycan_cache_mapping = mappings['glycan_mapping']
                logger.info("Loaded cached embedding mappings")
                return

        # Initialize preprocessor
        self.preprocessor = EmbeddingPreprocessor(
            cache_dir=str(self.cache_dir),
            protein_model=protein_model,
            glycan_method=glycan_method,
            glycan_vocab_path=glycan_vocab_path,
            glycan_hidden_dims=glycan_hidden_dims,
            glycan_readout=glycan_readout,
            device=self.device,
            **kwargs
        )

        # Get unique sequences from loaded data
        unique_proteins = self.data[self.sequence_col].unique().tolist()
        unique_glycans = self.glycan_columns

        logger.info(f"Found {len(unique_proteins)} unique proteins and {len(unique_glycans)} unique glycans")

        # Precompute embeddings
        protein_mapping = self.preprocessor.precompute_protein_embeddings(
            unique_proteins, force_recompute=force_recompute
        )

        glycan_mapping = self.preprocessor.precompute_glycan_embeddings(
            unique_glycans, force_recompute=force_recompute
        )

        self.protein_cache_mapping = protein_mapping
        self.glycan_cache_mapping = glycan_mapping

        # Save mappings for future use
        with open(mapping_file, 'wb') as f:
            pickle.dump({
                'protein_mapping': protein_mapping,
                'glycan_mapping': glycan_mapping,
                'glycan_params': {
                    'method': glycan_method,
                    'vocab_path': glycan_vocab_path,
                    'hidden_dims': glycan_hidden_dims,
                    'readout': glycan_readout
                }
            }, f)

        logger.info("Precomputed embeddings setup complete")

    def setup_clustering_splits(self,
                               save_splitter_path: Optional[str] = None,
                               plot_analysis: bool = False) -> Dict[str, List[str]]:
        """Setup clustering-based data splits (unchanged)"""
        if not self.use_clustering:
            logger.info("Clustering-based splits disabled")
            return self.split_by_protein(**self.clustering_params)

        if not self.use_precomputed or self.protein_cache_mapping is None:
            raise ValueError("Clustering requires precomputed protein embeddings. "
                           "Run setup_precomputed_embeddings() first.")

        logger.info("Setting up clustering-based splits...")

        # Get unique protein sequences and their embeddings
        unique_proteins = self.data[self.sequence_col].unique().tolist()

        # Ensure preprocessor is available for loading embeddings
        if self.preprocessor is None:
            # Create minimal preprocessor for loading embeddings
            self.preprocessor = EmbeddingPreprocessor(cache_dir=str(self.cache_dir))

        # Load protein embeddings
        protein_embeddings = self.preprocessor.load_cached_embeddings(
            unique_proteins, self.protein_cache_mapping
        )

        # Create clustered splits
        protein_splits, self.splitter = create_clustered_splits(
            protein_embeddings=protein_embeddings,
            protein_sequences=unique_proteins,
            plot_analysis=plot_analysis,
            save_splitter_path=save_splitter_path,
            **self.clustering_params
        )

        logger.info("Clustering-based splits setup complete")
        return protein_splits

    def split_by_protein(self,
                         test_size: float = 0.2,
                         val_size: float = 0.1,
                         random_state: int = 42,
                         **kwargs) -> Dict[str, List[str]]:
        """Split proteins for train/val/test (unchanged)"""
        from sklearn.model_selection import train_test_split

        unique_proteins = self.data[self.protein_col].unique()

        logger.info(f"Splitting {len(unique_proteins)} unique protein sequences:")
        logger.info(f"  Test size: {test_size:.1%}")
        logger.info(f"  Val size: {val_size:.1%}")
        logger.info(f"  Train size: {1 - test_size - val_size:.1%}")

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

    def create_pairs_dataset(self,
                           glycan_subset: Optional[List[str]] = None,
                           protein_subset: Optional[List[str]] = None,
                           max_pairs: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Create glycan-protein pairs and binding strengths (unchanged)"""
        if self.data is None:
            raise ValueError("Data not loaded.")

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

    def create_pytorch_dataloader(self,
                                protein_subset: Optional[List[str]] = None,
                                glycan_subset: Optional[List[str]] = None,
                                batch_size: int = 32,
                                shuffle: bool = True,
                                num_workers: int = 4,
                                normalize_targets: bool = True,
                                max_pairs: Optional[int] = None,
                                fusion_method: str = "concat",
                                cache_size_gb: float = 4.0,
                                preload_to_memory: bool = True,
                                persistent_workers: bool = True,
                                prefetch_factor: int = 2,
                                **kwargs) -> Tuple[DataLoader, Optional[torch.nn.Module]]:
        """
        Create PyTorch DataLoader with GPU optimization
        """
        # Create pairs and strengths
        pairs, strengths = self.create_pairs_dataset(
            glycan_subset=glycan_subset,
            protein_subset=protein_subset,
            max_pairs=max_pairs
        )

        logger.info(f"Processing {len(pairs)} pairs...")

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

        # Create GPU-optimized dataset
        if self.use_precomputed and self.protein_cache_mapping and self.glycan_cache_mapping:
            dataset = GPUOptimizedGlycanProteinDataset(
                pairs=pairs,
                targets=strengths_tensor,
                protein_cache_mapping=self.protein_cache_mapping,
                glycan_cache_mapping=self.glycan_cache_mapping,
                fusion_method=fusion_method,
                device=self.device,
                cache_size_gb=cache_size_gb,
                preload_to_memory=preload_to_memory
            )
            logger.info("Using GPU-optimized dataset with in-memory caching")
        else:
            # Fallback to original method
            from glycan_dataloader_cpu import CachedGlycanProteinDataset
            if self.embedder is None:
                raise ValueError("Embedder required when not using precomputed embeddings")

            cache_files = self._load_or_compute_embeddings_original(pairs, **kwargs)
            dataset = CachedGlycanProteinDataset(
                cache_files=cache_files,
                targets=strengths_tensor,
                device=self.device
            )
            logger.info("Using original cached embeddings dataset")

        # Create DataLoader with GPU optimization
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.startswith("cuda") else False,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            drop_last=False
        )

        logger.info(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}, "
                   f"num_workers={num_workers}")
        return dataloader, target_scaler

    def create_train_val_test_loaders(self,
                                    batch_size: int = 32,
                                    normalize_targets: bool = True,
                                    max_pairs_per_split: Optional[int] = None,
                                    fusion_method: str = "concat",
                                    # GPU optimization parameters
                                    num_workers: int = 4,
                                    cache_size_gb: float = 4.0,
                                    preload_to_memory: bool = True,
                                    persistent_workers: bool = True,
                                    prefetch_factor: int = 2,
                                    # Preprocessing parameters
                                    protein_model: str = "650M",
                                    glycan_method: str = "lstm",
                                    glycan_vocab_path: Optional[str] = None,
                                    glycan_hidden_dims: Optional[List[int]] = None,
                                    glycan_readout: str = "mean",
                                    force_recompute: bool = False,
                                    # Clustering parameters
                                    save_splitter_path: Optional[str] = None,
                                    plot_analysis: bool = False,
                                    **kwargs) -> Dict[str, DataLoader]:
        """
        Create train/validation/test DataLoaders with GPU optimization
        """
        # Setup precomputed embeddings if enabled
        if self.use_precomputed:
            preprocessing_kwargs = {
                'protein_model': protein_model,
                'glycan_method': glycan_method,
                'glycan_vocab_path': glycan_vocab_path,
                'glycan_hidden_dims': glycan_hidden_dims,
                'glycan_readout': glycan_readout,
                'force_recompute': force_recompute
            }
            self.setup_precomputed_embeddings(**preprocessing_kwargs)

        # Get protein splits (clustering or random)
        if self.use_clustering:
            clustering_kwargs = {
                'save_splitter_path': save_splitter_path,
                'plot_analysis': plot_analysis
            }
            protein_splits = self.setup_clustering_splits(**clustering_kwargs)
        else:
            protein_splits = self.split_by_protein(**self.clustering_params)

        dataloaders = {}
        target_scalers = {}

        # Create DataLoader for each split
        for split_name, proteins in protein_splits.items():
            if len(proteins) == 0:
                continue

            logger.info(f"Creating {split_name} DataLoader with {len(proteins)} proteins")

            # Prepare dataloader-specific kwargs
            dataloader_kwargs = {
                'protein_subset': proteins,
                'batch_size': batch_size,
                'shuffle': (split_name == 'train'),
                'normalize_targets': normalize_targets,
                'max_pairs': max_pairs_per_split,
                'fusion_method': fusion_method,
                'num_workers': num_workers,
                'cache_size_gb': cache_size_gb,
                'preload_to_memory': preload_to_memory,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor
            }

            # Add any additional kwargs that are relevant for dataloader creation
            for key, value in kwargs.items():
                if key not in dataloader_kwargs and key not in preprocessing_kwargs:
                    dataloader_kwargs[key] = value

            dataloader, scaler = self.create_pytorch_dataloader(**dataloader_kwargs)

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

    def _load_or_compute_embeddings_original(self, pairs, **kwargs):
        """Fallback to original embedding computation method"""
        from glycan_dataloader_cpu import GlycanProteinDataLoader

        original_loader = GlycanProteinDataLoader(
            embedder=self.embedder,
            cache_dir=str(self.cache_dir / "original_cache"),
            device=self.device
        )

        return original_loader._load_or_compute_embeddings(pairs, **kwargs)

    def get_cache_info(self) -> Dict:
        """Get information about the cache"""
        if self.preprocessor:
            return self.preprocessor.get_cache_info()
        else:
            # Fallback to original cache info
            cache_files = list(Path(self.cache_dir).glob("*.npy"))
            total_size = sum(f.stat().st_size for f in cache_files)
            return {
                "cached_files": len(cache_files),
                "cache_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir)
            }

    def clear_cache(self, cache_type: str = "all"):
        """Clear cache"""
        if self.preprocessor:
            self.preprocessor.clear_cache(cache_type)
        else:
            import shutil
            if Path(self.cache_dir).exists():
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(exist_ok=True)
                logger.info("Cleared cache directory")


# Enhanced convenience function
def create_enhanced_glycan_dataloaders(data_path: str = "data/v12_glycan_binding.csv",
                                     embedder=None,
                                     test_size: float = 0.2,
                                     val_size: float = 0.1,
                                     batch_size: int = 32,
                                     max_pairs: Optional[int] = None,
                                     device: Optional[str] = None,
                                     cache_dir: str = "preprocessed_embeddings",
                                     use_precomputed: bool = True,
                                     use_clustering: bool = True,
                                     n_clusters: int = 10,
                                     # GPU optimization parameters
                                     num_workers: int = 4,
                                     cache_size_gb: float = 4.0,
                                     preload_to_memory: bool = True,
                                     persistent_workers: bool = True,
                                     prefetch_factor: int = 2,
                                     # Embedding parameters
                                     protein_model: str = "650M",
                                     glycan_method: str = "lstm",
                                     glycan_vocab_path: Optional[str] = None,
                                     glycan_hidden_dims: Optional[List[int]] = None,
                                     glycan_readout: str = "mean",
                                     force_recompute: bool = False,
                                     save_splitter_path: Optional[str] = None,
                                     plot_analysis: bool = False,
                                     **kwargs) -> Dict[str, DataLoader]:
    """
    Enhanced convenience function with GPU optimization

    Args:
        data_path: Path to CSV/Excel data file
        embedder: GlycanProteinPairEmbedder instance (for backward compatibility)
        test_size: Test set fraction
        val_size: Validation set fraction
        batch_size: Batch size
        max_pairs: Maximum pairs per split
        device: Computing device
        cache_dir: Directory for caching embeddings
        use_precomputed: Whether to use precomputed embeddings
        use_clustering: Whether to use clustering-based splits
        n_clusters: Number of clusters for protein clustering
        num_workers: Number of data loading workers
        cache_size_gb: Size of in-memory cache in GB
        preload_to_memory: Whether to preload embeddings to memory
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch
        protein_model: ESM2 model size
        glycan_method: Glycan embedding method
        glycan_vocab_path: Path to glycan vocabulary
        glycan_hidden_dims: Hidden dimensions for glycan embedder
        glycan_readout: Readout function for graph-based methods
        force_recompute: Whether to recompute existing embeddings
        save_splitter_path: Path to save the fitted splitter
        plot_analysis: Whether to create clustering plots
        **kwargs: Additional arguments

    Returns:
        Dictionary with DataLoaders
    """

    clustering_params = {
        'n_clusters': n_clusters,
        'test_size': test_size,
        'val_size': val_size,
        'random_state': kwargs.get('random_state', 42),
        'use_pca': kwargs.get('use_pca', True)
    }

    loader = EnhancedGlycanProteinDataLoader(
        data_path=data_path,
        embedder=embedder,
        device=device,
        cache_dir=cache_dir,
        use_precomputed=use_precomputed,
        use_clustering=use_clustering,
        clustering_params=clustering_params
    )

    return loader.create_train_val_test_loaders(
        batch_size=batch_size,
        max_pairs_per_split=max_pairs,
        num_workers=num_workers,
        cache_size_gb=cache_size_gb,
        preload_to_memory=preload_to_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        protein_model=protein_model,
        glycan_method=glycan_method,
        glycan_vocab_path=glycan_vocab_path,
        glycan_hidden_dims=glycan_hidden_dims,
        glycan_readout=glycan_readout,
        force_recompute=force_recompute,
        save_splitter_path=save_splitter_path,
        plot_analysis=plot_analysis,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage with GPU optimization
    print("Testing GPU-Optimized Glycan PyTorch DataLoader")
    print("=" * 50)

    try:
        # Create GPU-optimized DataLoaders
        dataloaders = create_enhanced_glycan_dataloaders(
            data_path="data/v12_glycan_binding.csv",
            batch_size=64,  # Larger batch for GPU
            max_pairs=1000,  # For testing
            use_precomputed=True,
            use_clustering=True,
            n_clusters=5,
            # GPU optimization parameters
            num_workers=4,  # Multi-worker loading
            cache_size_gb=4.0,  # 4GB in-memory cache
            preload_to_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            # Embeddings
            protein_model="650M",
            glycan_method="lstm",
            glycan_vocab_path="GlycanEmbedder_Package/glycoword_vocab.pkl"
        )

        # Test training loop
        print("\nTesting GPU-optimized training loop:")
        train_loader = dataloaders['train']

        for batch_idx, (embeddings, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: embeddings {embeddings.shape}, targets {targets.shape}")
            print(f"  Data types: embeddings {embeddings.dtype}, targets {targets.dtype}")

            if batch_idx >= 2:
                break

        print("GPU-optimized DataLoader test completed successfully!")

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()