"""
Clustering-based Data Splitter
Implements protein sequence clustering for improved train/test/val splits
"""
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinClusteringSplitter:
    """
    Clustering-based data splitter for protein sequences
    Ensures better generalization by splitting based on protein structural similarity
    """

    def __init__(self,
                 n_clusters: int = 10,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 use_pca: bool = True,
                 pca_components: int = 50,
                 normalize_embeddings: bool = True):
        """
        Initialize the clustering splitter

        Args:
            n_clusters: Number of clusters for K-means
            test_size: Fraction of clusters for test set
            val_size: Fraction of clusters for validation set
            random_state: Random seed for reproducibility
            use_pca: Whether to use PCA before clustering
            pca_components: Number of PCA components
            normalize_embeddings: Whether to normalize embeddings before clustering
        """
        self.n_clusters = n_clusters
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.normalize_embeddings = normalize_embeddings

        # Initialize components
        self.kmeans = None
        self.scaler = None
        self.pca = None
        self.cluster_assignments = None
        self.split_assignments = None

        # Set random seed
        np.random.seed(random_state)

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Prepare embeddings for clustering (normalization + PCA)

        Args:
            embeddings: Protein embeddings array

        Returns:
            Processed embeddings ready for clustering
        """
        processed_embeddings = embeddings.copy()

        # Normalize embeddings
        if self.normalize_embeddings:
            if self.scaler is None:
                self.scaler = StandardScaler()
                processed_embeddings = self.scaler.fit_transform(processed_embeddings)
            else:
                processed_embeddings = self.scaler.transform(processed_embeddings)

            logger.info("Normalized embeddings using StandardScaler")

        # Apply PCA if requested
        if self.use_pca:
            n_components = min(self.pca_components, processed_embeddings.shape[1], processed_embeddings.shape[0])

            if self.pca is None:
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                processed_embeddings = self.pca.fit_transform(processed_embeddings)

                variance_explained = self.pca.explained_variance_ratio_.sum()
                logger.info(f"Applied PCA: {n_components} components, "
                            f"{variance_explained:.3f} variance explained")
            else:
                processed_embeddings = self.pca.transform(processed_embeddings)

        return processed_embeddings

    def fit_clusters(self,
                     protein_embeddings: np.ndarray,
                     protein_sequences: List[str]) -> Dict[str, int]:
        """
        Fit K-means clustering on protein embeddings

        Args:
            protein_embeddings: Array of protein embeddings
            protein_sequences: List of corresponding protein sequences

        Returns:
            Dictionary mapping protein sequences to cluster IDs
        """
        logger.info(f"Clustering {len(protein_sequences)} proteins into {self.n_clusters} clusters...")

        # Prepare embeddings
        processed_embeddings = self._prepare_embeddings(protein_embeddings)

        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(processed_embeddings)

        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(processed_embeddings, cluster_labels)
        logger.info(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")

        # Create sequence to cluster mapping
        self.cluster_assignments = {
            seq: int(label) for seq, label in zip(protein_sequences, cluster_labels)
        }

        # Log cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        logger.info("Cluster distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Cluster {label}: {count} proteins ({count / len(protein_sequences) * 100:.1f}%)")

        return self.cluster_assignments

    def assign_cluster_splits(self) -> Dict[int, str]:
        """
        Assign clusters to train/val/test splits

        Returns:
            Dictionary mapping cluster IDs to split names
        """
        if self.cluster_assignments is None:
            raise ValueError("Must fit clusters first using fit_clusters()")

        cluster_ids = list(range(self.n_clusters))
        np.random.shuffle(cluster_ids)

        # Calculate split sizes
        n_test_clusters = max(1, int(self.n_clusters * self.test_size))
        n_val_clusters = max(1, int(self.n_clusters * self.val_size))
        n_train_clusters = self.n_clusters - n_test_clusters - n_val_clusters

        # Assign splits
        test_clusters = cluster_ids[:n_test_clusters]
        val_clusters = cluster_ids[n_test_clusters:n_test_clusters + n_val_clusters]
        train_clusters = cluster_ids[n_test_clusters + n_val_clusters:]

        self.split_assignments = {}
        for cluster_id in test_clusters:
            self.split_assignments[cluster_id] = 'test'
        for cluster_id in val_clusters:
            self.split_assignments[cluster_id] = 'val'
        for cluster_id in train_clusters:
            self.split_assignments[cluster_id] = 'train'

        # Log split distribution
        logger.info("Cluster-to-split assignments:")
        logger.info(f"  Train clusters: {train_clusters} ({n_train_clusters} clusters)")
        logger.info(f"  Val clusters: {val_clusters} ({n_val_clusters} clusters)")
        logger.info(f"  Test clusters: {test_clusters} ({n_test_clusters} clusters)")

        return self.split_assignments

    def get_protein_splits(self) -> Dict[str, List[str]]:
        """
        Get protein sequences assigned to each split

        Returns:
            Dictionary with train/val/test protein lists
        """
        if self.cluster_assignments is None or self.split_assignments is None:
            raise ValueError("Must fit clusters and assign splits first")

        splits = {'train': [], 'val': [], 'test': []}

        for protein_seq, cluster_id in self.cluster_assignments.items():
            split_name = self.split_assignments[cluster_id]
            splits[split_name].append(protein_seq)

        # Log final split sizes
        logger.info("Final protein split distribution:")
        total_proteins = sum(len(proteins) for proteins in splits.values())
        for split_name, proteins in splits.items():
            logger.info(f"  {split_name}: {len(proteins)} proteins "
                        f"({len(proteins) / total_proteins * 100:.1f}%)")

        return splits

    def fit_and_split(self,
                      protein_embeddings: np.ndarray,
                      protein_sequences: List[str]) -> Dict[str, List[str]]:
        """
        Convenience method to fit clusters and create splits in one call

        Args:
            protein_embeddings: Array of protein embeddings
            protein_sequences: List of corresponding protein sequences

        Returns:
            Dictionary with train/val/test protein lists
        """
        self.fit_clusters(protein_embeddings, protein_sequences)
        self.assign_cluster_splits()
        return self.get_protein_splits()

    def plot_clustering_analysis(self,
                                 protein_embeddings: np.ndarray,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10)):
        """
        Create visualization of clustering results

        Args:
            protein_embeddings: Original protein embeddings
            save_path: Path to save the plot
            figsize: Figure size
        """
        if self.cluster_assignments is None:
            raise ValueError("Must fit clusters first")

        # Prepare embeddings for visualization
        processed_embeddings = self._prepare_embeddings(protein_embeddings)

        # Use 2D PCA for visualization
        pca_2d = PCA(n_components=2, random_state=self.random_state)
        embeddings_2d = pca_2d.fit_transform(processed_embeddings)

        # Get cluster labels
        cluster_labels = [self.cluster_assignments[seq] for seq in self.cluster_assignments.keys()]
        split_labels = [self.split_assignments[self.cluster_assignments[seq]]
                        for seq in self.cluster_assignments.keys()]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Protein Clustering Analysis', fontsize=16)

        # Plot 1: Clusters in 2D PCA space
        scatter1 = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=cluster_labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title('Protein Clusters (2D PCA)')
        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2f} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2f} variance)')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster ID')

        # Plot 2: Splits in 2D PCA space
        split_colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
        for split_name, color in split_colors.items():
            mask = np.array(split_labels) == split_name
            axes[0, 1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                               c=color, label=split_name, alpha=0.6)
        axes[0, 1].set_title('Train/Val/Test Splits')
        axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2f} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2f} variance)')
        axes[0, 1].legend()

        # Plot 3: Cluster size distribution
        cluster_sizes = [list(cluster_labels).count(i) for i in range(self.n_clusters)]
        axes[1, 0].bar(range(self.n_clusters), cluster_sizes)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Proteins')

        # Plot 4: Split size distribution
        split_sizes = [len([s for s in split_labels if s == split_name])
                       for split_name in ['train', 'val', 'test']]
        axes[1, 1].bar(['train', 'val', 'test'], split_sizes,
                       color=['blue', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_title('Split Size Distribution')
        axes[1, 1].set_ylabel('Number of Proteins')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Clustering analysis plot saved to: {save_path}")

        plt.show()

    def save_splitter(self, save_path: str):
        """Save the fitted splitter to disk"""
        save_data = {
            'n_clusters': self.n_clusters,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'normalize_embeddings': self.normalize_embeddings,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_assignments': self.cluster_assignments,
            'split_assignments': self.split_assignments
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Splitter saved to: {save_path}")

    @classmethod
    def load_splitter(cls, load_path: str) -> 'ProteinClusteringSplitter':
        """Load a fitted splitter from disk"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        # Create new instance
        splitter = cls(
            n_clusters=save_data['n_clusters'],
            test_size=save_data['test_size'],
            val_size=save_data['val_size'],
            random_state=save_data['random_state'],
            use_pca=save_data['use_pca'],
            pca_components=save_data['pca_components'],
            normalize_embeddings=save_data['normalize_embeddings']
        )

        # Restore fitted components
        splitter.kmeans = save_data['kmeans']
        splitter.scaler = save_data['scaler']
        splitter.pca = save_data['pca']
        splitter.cluster_assignments = save_data['cluster_assignments']
        splitter.split_assignments = save_data['split_assignments']

        logger.info(f"Splitter loaded from: {load_path}")
        return splitter


# Convenience function
def create_clustered_splits(protein_embeddings: np.ndarray,
                            protein_sequences: List[str],
                            n_clusters: int = 10,
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            random_state: int = 42,
                            use_pca: bool = True,
                            plot_analysis: bool = True,
                            save_splitter_path: Optional[str] = None) -> Tuple[
    Dict[str, List[str]], ProteinClusteringSplitter]:
    """
    Convenience function to create clustered data splits

    Args:
        protein_embeddings: Array of protein embeddings
        protein_sequences: List of protein sequences
        n_clusters: Number of clusters
        test_size: Test set fraction
        val_size: Validation set fraction
        random_state: Random seed
        use_pca: Whether to use PCA
        plot_analysis: Whether to create clustering plots
        save_splitter_path: Path to save the fitted splitter

    Returns:
        Tuple of (protein_splits, fitted_splitter)
    """
    # Create and fit splitter
    splitter = ProteinClusteringSplitter(
        n_clusters=n_clusters,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        use_pca=use_pca
    )

    protein_splits = splitter.fit_and_split(protein_embeddings, protein_sequences)

    # Optional visualization
    if plot_analysis:
        try:
            splitter.plot_clustering_analysis(protein_embeddings)
        except ImportError:
            logger.warning("Matplotlib not available, skipping clustering plots")

    # Optional save
    if save_splitter_path:
        splitter.save_splitter(save_splitter_path)

    return protein_splits, splitter


if __name__ == "__main__":
    # Example usage
    print("Testing protein clustering splitter...")

    # Generate sample data
    np.random.seed(42)
    n_proteins = 100
    embedding_dim = 1280

    # Create synthetic protein embeddings with some structure
    embeddings = np.random.randn(n_proteins, embedding_dim)
    sequences = [f"PROTEIN_{i}" for i in range(n_proteins)]

    # Test clustering
    protein_splits, splitter = create_clustered_splits(
        protein_embeddings=embeddings,
        protein_sequences=sequences,
        n_clusters=10,
        plot_analysis=True,
        save_splitter_path="test_splitter.pkl"
    )

    print("Clustering test completed successfully!")
    print(f"Split sizes: train={len(protein_splits['train'])}, "
          f"val={len(protein_splits['val'])}, test={len(protein_splits['test'])}")