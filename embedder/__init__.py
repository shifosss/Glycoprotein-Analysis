"""
Embedder module for glycan and protein sequence embedding.

This module provides tools for embedding glycan structures and protein sequences
using various methods including LSTM, GCN for glycans and ESM2 for proteins.
"""

# Import main components from submodules
from .Protein_Sequence_Embedder import ProteinEmbedderFactory
from .GlycanEmbedder_Package.glycan_embedder import GlycanEmbedder
from Integrated_Embedder import *

# Define what should be imported with "from embedder import *"
__all__ = [
    'ProteinEmbedderFactory',
    'GlycanEmbedder',
]
