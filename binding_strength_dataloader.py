"""
Glycan-Protein Binding Data Loader
Handles loading and preprocessing of glycan-protein binding data for machine learning
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import openpyxl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycanBindingDataLoader:
    """
    Data loader for glycan-protein binding datasets

    Handles Excel files with structure:
    - Rows: protein samples
    - Columns: glycan structures (with binding strength values) + protein info
    - Special columns: 'target' (protein sequence), 'protein' (protein identifier)
    """

    def __init__(self,
                 data_path: Optional[str] = None,
                 protein_col: str = 'protein',
                 sequence_col: str = 'target',
                 exclude_cols: Optional[List[str]] = None,
                 min_binding_threshold: Optional[float] = None,
                 max_binding_threshold: Optional[float] = None):
        """
        Initialize the data loader

        Args:
            data_path: Path to the Excel file
            protein_col: Column name containing protein identifiers
            sequence_col: Column name containing protein sequences
            exclude_cols: Additional columns to exclude from glycan data
            min_binding_threshold: Minimum binding strength threshold for filtering
            max_binding_threshold: Maximum binding strength threshold for filtering
        """
        self.data_path = data_path
        self.protein_col = protein_col
        self.sequence_col = sequence_col
        self.exclude_cols = exclude_cols or []
        self.min_binding_threshold = min_binding_threshold
        self.max_binding_threshold = max_binding_threshold

        # Data storage
        self.raw_data = None
        self.glycan_columns = None
        self.protein_data = None
        self.binding_data = None

        # Preprocessing objects
        self.binding_scaler = StandardScaler()
        self.protein_encoder = LabelEncoder()

        # Load data if path provided
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from Excel file

        Args:
            data_path: Path to the Excel file

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")

        try:
            # Try different engines for Excel reading
            if str(data_path).endswith('.xlsx'):
                self.raw_data = pd.read_excel(data_path, engine='openpyxl')
            elif str(data_path).endswith('.xls'):
                self.raw_data = pd.read_excel(data_path, engine='xlrd')
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            logger.info(f"Loaded data with shape: {self.raw_data.shape}")

            # Identify glycan columns (all columns except protein info columns)
            exclude_set = {self.protein_col, self.sequence_col} | set(self.exclude_cols)
            self.glycan_columns = [col for col in self.raw_data.columns if col not in exclude_set]

            logger.info(f"Identified {len(self.glycan_columns)} glycan columns")
            logger.info(f"Found {self.raw_data[self.protein_col].nunique()} unique proteins")

            # Extract protein and binding data
            self._extract_protein_data()
            self._extract_binding_data()

            return self.raw_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _extract_protein_data(self):
        """Extract protein information from the dataset"""
        self.protein_data = self.raw_data[[self.protein_col, self.sequence_col]].copy()

        # Check for missing sequences
        missing_sequences = self.protein_data[self.sequence_col].isna().sum()
        if missing_sequences > 0:
            logger.warning(f"Found {missing_sequences} missing protein sequences")
            # Remove rows with missing sequences
            self.protein_data = self.protein_data.dropna(subset=[self.sequence_col])
            self.raw_data = self.raw_data.dropna(subset=[self.sequence_col])

    def _extract_binding_data(self):
        """Extract and preprocess binding strength data"""
        self.binding_data = self.raw_data[self.glycan_columns].copy()

        # Check for missing values in binding data
        missing_values = self.binding_data.isna().sum().sum()
        if missing_values > 0:
            logger.info(f"Found {missing_values} missing binding values, filling with median")
            # Fill missing values with column median
            self.binding_data = self.binding_data.fillna(self.binding_data.median())

        # Apply binding thresholds if specified
        if self.min_binding_threshold is not None or self.max_binding_threshold is not None:
            self._apply_binding_thresholds()

        logger.info(f"Binding data statistics:")
        logger.info(f"  Shape: {self.binding_data.shape}")
        logger.info(f"  Min value: {self.binding_data.min().min():.3f}")
        logger.info(f"  Max value: {self.binding_data.max().max():.3f}")
        logger.info(f"  Mean value: {self.binding_data.mean().mean():.3f}")

    def _apply_binding_thresholds(self):
        """Apply binding strength thresholds"""
        original_shape = self.binding_data.shape

        if self.min_binding_threshold is not None:
            mask = (self.binding_data >= self.min_binding_threshold).any(axis=1)
            self.binding_data = self.binding_data[mask]
            self.protein_data = self.protein_data[mask]
            logger.info(f"Applied min threshold {self.min_binding_threshold}")

        if self.max_binding_threshold is not None:
            mask = (self.binding_data <= self.max_binding_threshold).any(axis=1)
            self.binding_data = self.binding_data[mask]
            self.protein_data = self.protein_data[mask]
            logger.info(f"Applied max threshold {self.max_binding_threshold}")

        if original_shape[0] != self.binding_data.shape[0]:
            logger.info(f"Filtered from {original_shape[0]} to {self.binding_data.shape[0]} samples")

    def get_pairs_and_strengths(self,
                                glycan_subset: Optional[List[str]] = None,
                                protein_subset: Optional[List[str]] = None) -> Tuple[
        List[Tuple[str, str]], List[float]]:
        """
        Get glycan-protein pairs and their binding strengths

        Args:
            glycan_subset: List of specific glycan structures to include
            protein_subset: List of specific protein names to include

        Returns:
            Tuple of (pairs, strengths) where pairs are (glycan_iupac, protein_sequence)
        """
        if self.binding_data is None or self.protein_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Filter glycans if subset specified
        glycans_to_use = glycan_subset if glycan_subset else self.glycan_columns

        # Filter proteins if subset specified
        if protein_subset:
            protein_mask = self.protein_data[self.protein_col].isin(protein_subset)
            protein_data = self.protein_data[protein_mask]
            binding_data = self.binding_data[protein_mask]
        else:
            protein_data = self.protein_data
            binding_data = self.binding_data

        pairs = []
        strengths = []

        # Create all glycan-protein pairs
        for idx, (_, protein_row) in enumerate(protein_data.iterrows()):
            protein_sequence = protein_row[self.sequence_col]

            for glycan in glycans_to_use:
                if glycan in binding_data.columns:
                    binding_strength = binding_data.iloc[idx][glycan]
                    pairs.append((glycan, protein_sequence))
                    strengths.append(binding_strength)

        logger.info(f"Generated {len(pairs)} glycan-protein pairs")
        return pairs, strengths

    def get_top_binding_pairs(self,
                              n_top: int = 1000,
                              by_glycan: bool = True) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Get top binding pairs by strength

        Args:
            n_top: Number of top pairs to return
            by_glycan: If True, get top pairs for each glycan; if False, get overall top pairs

        Returns:
            Tuple of (pairs, strengths)
        """
        pairs, strengths = self.get_pairs_and_strengths()

        if by_glycan:
            # Get top binding pairs for each glycan
            top_pairs = []
            top_strengths = []

            glycan_dict = {}
            for i, (glycan, protein) in enumerate(pairs):
                if glycan not in glycan_dict:
                    glycan_dict[glycan] = []
                glycan_dict[glycan].append((i, strengths[i]))

            pairs_per_glycan = max(1, n_top // len(self.glycan_columns))

            for glycan, strength_list in glycan_dict.items():
                # Sort by binding strength (descending)
                strength_list.sort(key=lambda x: x[1], reverse=True)

                for i, strength in strength_list[:pairs_per_glycan]:
                    top_pairs.append(pairs[i])
                    top_strengths.append(strength)

        else:
            # Get overall top pairs
            sorted_indices = np.argsort(strengths)[::-1]  # Descending order
            top_indices = sorted_indices[:n_top]

            top_pairs = [pairs[i] for i in top_indices]
            top_strengths = [strengths[i] for i in top_indices]

        logger.info(f"Selected {len(top_pairs)} top binding pairs")
        return top_pairs, top_strengths

    def split_by_protein(self,
                         test_size: float = 0.2,
                         val_size: float = 0.1,
                         random_state: int = 42) -> Dict[str, Tuple[List[Tuple[str, str]], List[float]]]:
        """
        Split data by protein (ensures no protein leakage between sets)

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (pairs, strengths)
        """
        unique_proteins = self.protein_data[self.protein_col].unique()

        # Split proteins into train/val/test
        train_proteins, test_proteins = train_test_split(
            unique_proteins, test_size=test_size, random_state=random_state
        )

        if val_size > 0:
            train_proteins, val_proteins = train_test_split(
                train_proteins, test_size=val_size / (1 - test_size), random_state=random_state
            )
        else:
            val_proteins = []

        # Get pairs for each split
        splits = {}

        splits['train'] = self.get_pairs_and_strengths(protein_subset=train_proteins.tolist())
        if val_size > 0:
            splits['val'] = self.get_pairs_and_strengths(protein_subset=val_proteins.tolist())
        splits['test'] = self.get_pairs_and_strengths(protein_subset=test_proteins.tolist())

        logger.info(f"Split sizes - Train: {len(splits['train'][0])}, "
                    f"Val: {len(splits.get('val', [[], []])[0])}, "
                    f"Test: {len(splits['test'][0])}")

        return splits

    def get_glycan_statistics(self) -> pd.DataFrame:
        """Get statistics for each glycan"""
        if self.binding_data is None:
            raise ValueError("Data not loaded")

        stats = []
        for glycan in self.glycan_columns:
            values = self.binding_data[glycan]
            stats.append({
                'glycan': glycan,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'count': len(values),
                'positive_count': (values > 0).sum(),
                'negative_count': (values < 0).sum()
            })

        return pd.DataFrame(stats).sort_values('mean', ascending=False)

    def get_protein_statistics(self) -> pd.DataFrame:
        """Get statistics for each protein"""
        if self.protein_data is None or self.binding_data is None:
            raise ValueError("Data not loaded")

        stats = []
        for protein in self.protein_data[self.protein_col].unique():
            protein_mask = self.protein_data[self.protein_col] == protein
            protein_bindings = self.binding_data[protein_mask]

            if len(protein_bindings) > 0:
                values = protein_bindings.values.flatten()
                sequence = self.protein_data[protein_mask][self.sequence_col].iloc[0]

                stats.append({
                    'protein': protein,
                    'sequence_length': len(sequence),
                    'n_samples': len(protein_bindings),
                    'mean_binding': values.mean(),
                    'std_binding': values.std(),
                    'min_binding': values.min(),
                    'max_binding': values.max(),
                    'positive_bindings': (values > 0).sum(),
                    'negative_bindings': (values < 0).sum()
                })

        return pd.DataFrame(stats).sort_values('mean_binding', ascending=False)

    def filter_glycans_by_variance(self, min_variance: float = 0.01) -> List[str]:
        """
        Filter glycans by variance to remove low-variance features

        Args:
            min_variance: Minimum variance threshold

        Returns:
            List of glycans that pass the variance filter
        """
        if self.binding_data is None:
            raise ValueError("Data not loaded")

        variances = self.binding_data.var()
        high_variance_glycans = variances[variances >= min_variance].index.tolist()

        logger.info(f"Filtered from {len(self.glycan_columns)} to {len(high_variance_glycans)} glycans "
                    f"(min variance: {min_variance})")

        return high_variance_glycans

    def save_processed_data(self, save_path: str):
        """Save processed data for later use"""
        data_dict = {
            'binding_data': self.binding_data,
            'protein_data': self.protein_data,
            'glycan_columns': self.glycan_columns,
            'metadata': {
                'protein_col': self.protein_col,
                'sequence_col': self.sequence_col,
                'n_proteins': self.protein_data[self.protein_col].nunique(),
                'n_glycans': len(self.glycan_columns),
                'n_samples': len(self.protein_data)
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)

        logger.info(f"Saved processed data to {save_path}")

    def load_processed_data(self, load_path: str):
        """Load previously processed data"""
        with open(load_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.binding_data = data_dict['binding_data']
        self.protein_data = data_dict['protein_data']
        self.glycan_columns = data_dict['glycan_columns']

        logger.info(f"Loaded processed data from {load_path}")
        logger.info(f"Data shape: {self.binding_data.shape}")


# Convenience function for quick loading
def load_glycan_binding_data(data_path: str, **kwargs) -> GlycanBindingDataLoader:
    """
    Quick function to load glycan binding data

    Args:
        data_path: Path to the Excel file
        **kwargs: Additional arguments for GlycanBindingDataLoader

    Returns:
        Loaded data loader
    """
    loader = GlycanBindingDataLoader(data_path=data_path, **kwargs)
    return loader


if __name__ == "__main__":
    # Example usage
    print("Example usage of GlycanBindingDataLoader")

    # Load data
    data_path = "glycan_binding_example v12.xlsx"  # Update with your file path

    try:
        loader = GlycanBindingDataLoader(data_path=data_path)

        # Get basic statistics
        print(f"\nDataset Overview:")
        print(f"- {len(loader.protein_data)} protein samples")
        print(f"- {len(loader.glycan_columns)} glycan structures")
        print(f"- {loader.protein_data[loader.protein_col].nunique()} unique proteins")

        # Get all pairs and strengths
        pairs, strengths = loader.get_pairs_and_strengths()
        print(f"\nGenerated {len(pairs)} total glycan-protein pairs")
        print(f"Binding strength range: {min(strengths):.3f} to {max(strengths):.3f}")

        # Split by protein
        splits = loader.split_by_protein(test_size=0.2, val_size=0.1)
        print(f"\nData splits:")
        print(f"- Train: {len(splits['train'][0])} pairs")
        print(f"- Val: {len(splits['val'][0])} pairs")
        print(f"- Test: {len(splits['test'][0])} pairs")

        # Get top binding pairs
        top_pairs, top_strengths = loader.get_top_binding_pairs(n_top=100)
        print(f"\nTop 100 binding pairs:")
        print(f"- Highest strength: {max(top_strengths):.3f}")
        print(f"- Average of top: {np.mean(top_strengths):.3f}")

        # Get glycan statistics
        glycan_stats = loader.get_glycan_statistics()
        print(f"\nTop 5 glycans by average binding strength:")
        print(glycan_stats.head()[['glycan', 'mean', 'std']].to_string(index=False))

        print("\nDataLoader created successfully!")

    except Exception as e:
        print(f"Error in example: {e}")
        print("Make sure the data file exists and has the correct format")