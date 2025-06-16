"""
GlycanEmbedder: Standalone glycan embedding tool.

Usage:
    from glycan_embedder import embed_glycans, GlycanEmbedder
    
    # Quick usage
    embeddings = embed_glycans(glycan_list, method='graph', embedding_dim=128)
    
    # Advanced usage
    embedder = GlycanEmbedder()
    embeddings = embedder.embed_glycans(glycan_list, method='sequence')
"""

from .glycan_embedder import GlycanEmbedder, embed_glycans

__version__ = "1.0.0"
__all__ = ["GlycanEmbedder", "embed_glycans"]
