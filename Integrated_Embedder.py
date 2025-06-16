"""
Glycan-Protein Pair Embedder
Combines glycan and protein embeddings for downstream model training
Updated to work with refactored GlycanEmbedder
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
import logging

# Import the embedders
from Protein_Sequence_Embedder import ProteinEmbedderFactory
from GlycanEmbedder_Package.glycan_embedder import GlycanEmbedder, GlycanGCN, GlycanLSTM, GlycanBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlycanProteinPairEmbedder:
    """
    Embedder for glycan-protein pairs with concatenation and attention-based fusion
    Updated to work with refactored GlycanEmbedder
    """

    def __init__(self,
                 # Protein embedder settings
                 protein_model: str = "650M",
                 protein_model_dir: str = "resources/esm-model-weights",

                 # Glycan embedder settings
                 glycan_method: str = "lstm",
                 glycan_vocab_path: Optional[str] = None,
                 glycan_hidden_dims: Optional[List[int]] = None,
                 glycan_readout: str = "mean",
                 glycan_custom_params: Optional[dict] = None,

                 # Fusion settings
                 fusion_method: str = "concat",  # "concat" or "attention"

                 # Device settings
                 device: Optional[str] = None):
        """
        Initialize the glycan-protein pair embedder

        Args:
            protein_model: ESM2 model size ("650M" or "3B")
            protein_model_dir: Directory for protein model weights
            glycan_method: Glycan embedding method (gcn, lstm, bert, etc.)
            glycan_vocab_path: Path to glycan vocabulary file
            glycan_hidden_dims: Hidden dimensions for glycan embedder
            glycan_readout: Readout function for graph-based methods
            glycan_custom_params: Additional parameters for glycan embedder
            fusion_method: "concat" for concatenation, "attention" for attention-based fusion
            device: Device to use (None = auto-detect)
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize protein embedder
        logger.info(f"Initializing protein embedder ({protein_model})...")
        self.protein_embedder = ProteinEmbedderFactory.create_embedder(
            "esm2",
            model_name=protein_model,
            model_dir=protein_model_dir,
            device=self.device
        )
        self.protein_dim = self.protein_embedder.get_embedding_dim()

        # Initialize glycan embedder with matching dimension
        logger.info(f"Initializing glycan embedder ({glycan_method}) with dim={self.protein_dim}...")
        self.glycan_embedder = GlycanEmbedder(
            vocab_path=glycan_vocab_path,
            device=self.device
        )
        self.glycan_method = glycan_method
        self.glycan_dim = self.protein_dim  # Match protein dimension

        # Store glycan embedder parameters
        self.glycan_hidden_dims = glycan_hidden_dims
        self.glycan_readout = glycan_readout
        self.glycan_custom_params = glycan_custom_params or {}

        # Fusion settings
        self.fusion_method = fusion_method

        # Calculate output dimension
        if fusion_method == "concat":
            self.output_dim = self.protein_dim * 2
        else:  # attention
            self.output_dim = self.protein_dim
            self._init_attention_fusion()

        logger.info(f"Embedder initialized: fusion={fusion_method}, output_dim={self.output_dim}")

    def _init_attention_fusion(self):
        """Initialize attention-based fusion components"""
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.protein_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        ).to(self.device)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.protein_dim * 2, self.protein_dim),
            nn.LayerNorm(self.protein_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.device)

    def embed_pairs(self,
                    pairs: List[Tuple[str, str]],
                    batch_size: int = 32,
                    return_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Embed glycan-protein pairs

        Args:
            pairs: List of (glycan_iupac, protein_sequence) tuples
            batch_size: Process in batches
            return_numpy: If True, return numpy array; else return torch tensor

        Returns:
            Embeddings of shape (n_pairs, output_dim)
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # Separate glycans and proteins
            glycans = [pair[0] for pair in batch_pairs]
            proteins = [pair[1] for pair in batch_pairs]

            # Get embeddings
            with torch.no_grad():
                # Prepare glycan embedder parameters
                glycan_params = {
                    'method': self.glycan_method,
                    'embedding_dim': self.glycan_dim
                }

                # Add hidden dimensions if specified
                if self.glycan_hidden_dims:
                    glycan_params['hidden_dims'] = self.glycan_hidden_dims

                # Add readout for graph methods
                if self.glycan_method in ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn']:
                    glycan_params['readout'] = self.glycan_readout

                # Add any custom parameters
                glycan_params.update(self.glycan_custom_params)

                # Embed glycans with refactored embedder
                glycan_emb = self.glycan_embedder.embed_glycans(
                    glycans,
                    **glycan_params
                )

                logger.debug(f"Glycan embeddings shape: {glycan_emb.shape}")

                # Embed proteins
                protein_emb = torch.from_numpy(
                    self.protein_embedder.embed(proteins)
                ).to(self.device)

                logger.debug(f"Protein embeddings shape: {protein_emb.shape}")

                # Normalize embeddings
                glycan_emb = nn.functional.normalize(glycan_emb, p=2, dim=1)
                protein_emb = nn.functional.normalize(protein_emb, p=2, dim=1)

                # Combine embeddings
                if self.fusion_method == "concat":
                    combined = torch.cat([glycan_emb, protein_emb], dim=1)
                else:  # attention
                    combined = self._attention_fusion(glycan_emb, protein_emb)

                logger.debug(f"Combined shape: {combined.shape}")

                all_embeddings.append(combined)

        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)

        if return_numpy:
            return final_embeddings.cpu().numpy()
        else:
            return final_embeddings

    def _attention_fusion(self, glycan_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        """Apply attention-based fusion"""
        # Expand dimensions for attention
        glycan_exp = glycan_emb.unsqueeze(1)  # [batch, 1, dim]
        protein_exp = protein_emb.unsqueeze(1)  # [batch, 1, dim]

        # Cross-attention: glycan attends to protein
        attended, _ = self.cross_attention(glycan_exp, protein_exp, protein_exp)
        attended = attended.squeeze(1)  # [batch, dim]

        # Combine attended glycan with protein
        combined = torch.cat([attended, protein_emb], dim=1)  # [batch, dim*2]

        # Final fusion
        output = self.fusion_layer(combined)  # [batch, dim]

        return output

    def get_output_dim(self) -> int:
        """Get the dimension of the output embeddings"""
        return self.output_dim

    def save_attention_weights(self, path: str):
        """Save attention fusion weights (only for attention fusion)"""
        if self.fusion_method == "attention":
            state = {
                'cross_attention': self.cross_attention.state_dict(),
                'fusion_layer': self.fusion_layer.state_dict()
            }
            torch.save(state, path)
            logger.info(f"Saved attention weights to {path}")

    def load_attention_weights(self, path: str):
        """Load attention fusion weights (only for attention fusion)"""
        if self.fusion_method == "attention":
            state = torch.load(path, map_location=self.device)
            self.cross_attention.load_state_dict(state['cross_attention'])
            self.fusion_layer.load_state_dict(state['fusion_layer'])
            logger.info(f"Loaded attention weights from {path}")

    def create_custom_glycan_embedder(self, embedder_class, **kwargs):
        """
        Create a custom glycan embedder using specific embedder class

        Args:
            embedder_class: One of the glycan embedder classes (e.g., GlycanGCN, GlycanLSTM)
            **kwargs: Arguments for the embedder class

        Returns:
            Custom embedder instance
        """
        # Get vocabulary info from the main embedder
        num_units = len(self.glycan_embedder.units)
        num_relations = len(self.glycan_embedder.links)
        glycoword_dim = len(self.glycan_embedder.glycowords)

        # Add vocabulary-specific parameters based on embedder type
        if embedder_class in [GlycanGCN, GlycanLSTM, GlycanBERT]:
            if hasattr(embedder_class, '__name__'):
                class_name = embedder_class.__name__
                if 'GCN' in class_name or 'GAT' in class_name or 'GIN' in class_name:
                    kwargs['num_unit'] = num_units
                elif 'RGCN' in class_name or 'CompGCN' in class_name:
                    kwargs['num_unit'] = num_units
                    kwargs['num_relation'] = num_relations
                elif 'LSTM' in class_name or 'CNN' in class_name or 'ResNet' in class_name or 'BERT' in class_name:
                    kwargs['glycoword_dim'] = glycoword_dim

        return embedder_class(**kwargs).to(self.device)


# Convenience function
def embed_glycan_protein_pairs(pairs: List[Tuple[str, str]],
                               protein_model: str = "650M",
                               glycan_method: str = "lstm",
                               glycan_hidden_dims: Optional[List[int]] = None,
                               glycan_readout: str = "mean",
                               fusion_method: str = "concat",
                               batch_size: int = 32,
                               **kwargs) -> np.ndarray:
    """
    Quick function to embed glycan-protein pairs with refactored embedder

    Args:
        pairs: List of (glycan_iupac, protein_sequence) tuples
        protein_model: ESM2 model ("650M" or "3B")
        glycan_method: Glycan embedding method
        glycan_hidden_dims: Hidden dimensions for glycan embedder
        glycan_readout: Readout function for graph methods
        fusion_method: "concat" or "attention"
        batch_size: Batch size for processing
        **kwargs: Additional arguments for the embedder

    Returns:
        np.ndarray: Combined embeddings of shape (n_pairs, output_dim)
    """
    # Extract glycan-specific parameters
    glycan_custom_params = {}
    glycan_param_keys = ['short_cut', 'batch_norm', 'activation', 'concat_hidden',
                         'num_heads', 'num_layers', 'dropout', 'kernel_size',
                         'num_blocks', 'bidirectional', 'eps', 'learn_eps']

    for key in glycan_param_keys:
        if key in kwargs:
            glycan_custom_params[key] = kwargs.pop(key)

    embedder = GlycanProteinPairEmbedder(
        protein_model=protein_model,
        glycan_method=glycan_method,
        glycan_hidden_dims=glycan_hidden_dims,
        glycan_readout=glycan_readout,
        glycan_custom_params=glycan_custom_params,
        fusion_method=fusion_method,
        **kwargs
    )

    return embedder.embed_pairs(pairs, batch_size=batch_size, return_numpy=True)


if __name__ == "__main__":
    vocab = "GlycanEmbedder_Package/glycoword_vocab.pkl"
    # Example usage with refactored embedder
    pairs = [
        ("Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
         "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("Neu5Ac(a2-3)Gal(b1-4)Glc",
         "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP")
    ]

    # # Example 1: Simple concatenation with default settings
    # print("Example 1: Concatenation fusion with default LSTM")
    # embeddings_concat = embed_glycan_protein_pairs(
    #     pairs,
    #     protein_model="650M",
    #     protein_model_dir="resources/esm-model-weights",
    #     glycan_method="lstm",
    #     glycan_vocab_path=vocab,
    #     fusion_method="concat"
    # )
    # print(f"Concatenation output shape: {embeddings_concat.shape}")
    # print(f"First embedding (first 10 values): {embeddings_concat[0, :10]}")
    #
    # # Example 2: Attention fusion with custom GCN
    # print("\nExample 2: Attention fusion with custom GCN")
    # embeddings_attention = embed_glycan_protein_pairs(
    #     pairs,
    #     protein_model="650M",
    #     protein_model_dir="resources/esm-model-weights",
    #     glycan_method="gcn",
    #     glycan_hidden_dims=[512, 256, 640],  # End with protein dimension
    #     glycan_readout="dual",  # Use dual readout (mean + max)
    #     glycan_vocab_path=vocab,
    #     fusion_method="attention",
    #     short_cut=True,
    #     batch_norm=True
    # )
    # print(f"Attention output shape: {embeddings_attention.shape}")
    # print(f"First embedding (first 50 values): {embeddings_attention[0, :50]}")
    #
    # # Example 3: Using with PyTorch model and BERT glycan embedder
    # print("\nExample 3: Integration with PyTorch model (BERT glycan embedder)")
    # embedder = GlycanProteinPairEmbedder(
    #     protein_model="650M",
    #     protein_model_dir="resources/esm-model-weights",
    #     glycan_method="bert",
    #     glycan_hidden_dims=None,  # BERT doesn't use hidden_dims
    #     fusion_method="attention",
    #     glycan_vocab_path=vocab,
    #     glycan_custom_params={
    #         'num_layers': 8,
    #         'num_heads': 16,
    #         'intermediate_dim': 3072,
    #         'hidden_dropout': 0.1
    #     }
    # )
    #
    # # Get embeddings as torch tensor for direct use in models
    # embeddings_torch = embedder.embed_pairs(pairs, return_numpy=False)
    # print(f"Torch tensor shape: {embeddings_torch.shape}")
    # print(f"Device: {embeddings_torch.device}")
    #
    # # Example 4: Using custom embedder class directly
    # print("\nExample 4: Using custom embedder class directly")
    # custom_embedder = GlycanProteinPairEmbedder(
    #     protein_model="650M",
    #     protein_model_dir="resources/esm-model-weights",
    #     glycan_method="gin",  # Using GIN
    #     glycan_vocab_path=vocab,
    #     fusion_method="concat"
    # )
    #
    # # Create a custom GIN embedder with specific architecture
    # from GlycanEmbedder_Package.glycan_embedder import GlycanGIN
    #
    # custom_gin = custom_embedder.create_custom_glycan_embedder(
    #     GlycanGIN,
    #     input_dim=256,
    #     hidden_dims=[512, 512, 1280],
    #     num_mlp_layer=3,
    #     eps=0.1,
    #     learn_eps=True,
    #     readout="attention"
    # )
    # print(f"Custom GIN output dimension: {custom_gin.get_output_dim()}")
    #
    # # Example 5: Comparing different glycan methods
    # print("\nExample 5: Comparing different glycan embedding methods")
    # methods_to_compare = ['gcn', 'lstm', 'cnn', 'bert']
    #
    # for method in methods_to_compare:
    #     method_embeddings = embed_glycan_protein_pairs(
    #         pairs[:1],  # Just first pair for speed
    #         protein_model="650M",
    #         protein_model_dir="resources/esm-model-weights",
    #         glycan_method=method,
    #         glycan_vocab_path=vocab,
    #         fusion_method="concat"
    #     )
    #     print(f"  {method.upper()}: embedding shape = {method_embeddings.shape}")
    #     embeddings_torch = embedder.embed_pairs(pairs, return_numpy=False)
    #     print(f"Torch tensor shape: {embeddings_torch.shape}")
    #     print(f"Device: {embeddings_torch.device}")
    #
    # # Example 6
    # print("\nExample 6")
    # custom_embedder = embed_glycan_protein_pairs(
    #     pairs[:1],  # Just first pair for speed
    #     protein_model="650M",
    #     protein_model_dir="resources/esm-model-weights",
    #     glycan_method="rgcn",  # Using RGCN
    #     glycan_hidden_dims=[1024, 768, 384, 1280],  # End with protein dimension
    #     glycan_readout="mean",  # Use dual readout (mean + max)
    #     glycan_vocab_path=vocab,
    #     fusion_method="attention",
    #     short_cut=False,
    #     batch_norm=False,
    #     concat_hidden=False
    # )
    # print(f"Concatenation output shape: {custom_embedder.shape}")
    # print(f"First embedding (first 10 values): {custom_embedder[0, :10]}")

    # Example 7 Comparison between attention and concat
    print("Example 7: Part 1 CNN + Concat")
    embeddings_concat = embed_glycan_protein_pairs(
        pairs[:1],
        protein_model="650M",
        protein_model_dir="resources/esm-model-weights",
        glycan_method="cnn",
        glycan_hidden_dims=[512, 256, 1280],  # End with protein dimension
        glycan_readout="dual",  # Use dual readout (mean + max)
        glycan_vocab_path=vocab,
        fusion_method="concat"
    )
    print(f"Concatenation output shape: {embeddings_concat.shape}")
    print(f"First embedding (first 50 values): {embeddings_concat[0, :50]}")

    print("Example 7: Part 2 CNN + Attention")
    embeddings_concat = embed_glycan_protein_pairs(
        pairs[:1],
        protein_model="650M",
        protein_model_dir="resources/esm-model-weights",
        glycan_method="cnn",
        glycan_hidden_dims=[512, 256, 1280],  # End with protein dimension
        glycan_readout="dual",  # Use dual readout (mean + max)
        glycan_vocab_path=vocab,
        fusion_method="attention"
    )
    print(f"Concatenation output shape: {embeddings_concat.shape}")
    print(f"Second embedding (first 50 values): {embeddings_concat[0, :50]}")