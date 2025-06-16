# Protein Sequence Embedder

A clean and extensible Python library for embedding protein sequences using state-of-the-art models like ESM-2.

## Features

- 🧬 Support for ESM-2 models (650M and 3B parameters)
- 🚀 Automatic model downloading and caching
- 🔧 Simple and extensible architecture for adding new models
- 💾 Efficient batch processing
- 🖥️ GPU support with automatic device detection

## Installation

```bash
# Install required dependencies
pip install torch fair-esm numpy

# Clone or download the protein_embedder.py file
```

## Quick Start

```python
from protein_embedder import embed_proteins

# Single sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
embedding = embed_proteins(sequence)
print(f"Embedding shape: {embedding.shape}")  # (1, 1280) for 650M model

# Multiple sequences with custom model directory
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
]
embeddings = embed_proteins(sequences, model_dir="resource/esm-model-weights")
print(f"Embeddings shape: {embeddings.shape}")  # (2, 1280) for 650M model
```

## Detailed Usage

### Using Different Models

```python
from protein_embedder import embed_proteins

# Use ESM-2 650M model (default)
embeddings_650m = embed_proteins(sequences, model="650M")
print(f"650M embeddings shape: {embeddings_650m.shape}")  # (n_sequences, 1280)

# Use ESM-2 3B model (requires more memory)
embeddings_3b = embed_proteins(sequences, model="3B")
print(f"3B embeddings shape: {embeddings_3b.shape}")  # (n_sequences, 2560)
```

### Advanced Usage with Class Interface

```python
from protein_embedder import ESM2Embedder

# Initialize embedder with custom settings
embedder = ESM2Embedder(
    model_name="650M",           # or "3B"
    model_dir="./my_models",     # custom model directory
    device="cuda",               # force GPU usage
    repr_layer=-1                # use last layer (default)
)

# Get embedding dimension
dim = embedder.get_embedding_dim()
print(f"Embedding dimension: {dim}")

# Embed sequences
embeddings = embedder.embed(sequences)
```

### Using Factory Pattern

```python
from protein_embedder import ProteinEmbedderFactory

# Create embedder using factory
embedder = ProteinEmbedderFactory.create_embedder(
    embedder_type="esm2",
    model_name="650M"
)

embeddings = embedder.embed(sequences)
```

## Adding New Protein Embedding Models

The library is designed to be easily extensible. Here's how to add a new protein embedding model:

### Step 1: Create Your Embedder Class

```python
from protein_embedder import ProteinEmbedder, ProteinEmbedderFactory
import numpy as np

class MyCustomEmbedder(ProteinEmbedder):
    def __init__(self, model_path: str, **kwargs):
        # Initialize your model here
        self.model = load_my_model(model_path)
        self.embedding_dim = 768  # example dimension
    
    def embed(self, sequences):
        # Convert single sequence to list
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Your embedding logic here
        embeddings = []
        for seq in sequences:
            embedding = self.model.encode(seq)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_embedding_dim(self):
        return self.embedding_dim
```

### Step 2: Register Your Embedder

```python
# Register the new embedder
ProteinEmbedderFactory.register_embedder("my_model", MyCustomEmbedder)

# Now you can use it
embeddings = embed_proteins(
    sequences, 
    embedder_type="my_model",
    model_path="path/to/model"
)
```

## API Reference

### `embed_proteins(sequences, model="650M", embedder_type="esm2", **kwargs)`

Convenience function to embed protein sequences.

**Parameters:**
- `sequences`: Single sequence (str) or list of sequences (List[str])
- `model`: Model size for ESM2 - "650M" or "3B" (default: "650M")
- `embedder_type`: Type of embedder to use (default: "esm2")
- `**kwargs`: Additional arguments passed to the embedder, including:
  - `model_dir`: Directory to store downloaded models (default: "./models")
  - `device`: Device to run on - "cuda" or "cpu" (default: auto-detect)
  - `repr_layer`: Which layer to extract representations from (default: -1)

**Returns:**
- `np.ndarray`: Embedding matrix of shape (n_sequences, embedding_dim)

### `ProteinEmbedder` (Abstract Base Class)

**Methods:**
- `embed(sequences)`: Embed protein sequences into vectors
- `get_embedding_dim()`: Get the dimension of the embeddings

### `ESM2Embedder`

**Parameters:**
- `model_name`: "650M" or "3B" (default: "650M")
- `model_dir`: Directory to store models (default: "./models")
- `device`: Device to run on - "cuda" or "cpu" (default: auto-detect)
- `repr_layer`: Which layer to extract representations from (default: -1 for last layer)

### `ProteinEmbedderFactory`

**Class Methods:**
- `create_embedder(embedder_type, **kwargs)`: Create a protein embedder
- `register_embedder(name, embedder_class)`: Register a new embedder type

## Model Information

### ESM-2 Models

| Model | Parameters | Embedding Dimension | Memory Usage |
|-------|------------|-------------------|--------------|
| ESM-2 650M | 650M | 1280 | ~2.5 GB |
| ESM-2 3B | 3B | 2560 | ~12 GB |

Both models are trained on UniRef50 and provide state-of-the-art protein representations.

### Model Storage

By default, models are stored in `./models`. You can specify a custom directory:

```python
# Using the convenience function
embeddings = embed_proteins(sequences, model_dir="resource/esm-model-weights")

# Using the class interface
embedder = ESM2Embedder(model_name="650M", model_dir="resource/esm-model-weights")
```

The models will be downloaded directly to your specified directory on first use, avoiding the default PyTorch hub cache location.

## Notes

- Models are automatically downloaded on first use and cached in the model directory
- The embeddings are computed by averaging over the sequence length (mean pooling)
- GPU is automatically used if available, but can be manually specified
- For long sequences or large batches, consider using GPU for faster processing

## Troubleshooting

**Out of Memory Error with 3B Model:**
- The 3B model requires significant GPU memory (~12GB)
- Try using CPU: `ESM2Embedder(model_name="3B", device="cpu")`
- Or use the smaller 650M model

**Model Download Issues:**
- Check your internet connection
- Ensure the model directory has write permissions
- Models are downloaded from Facebook's servers

**Import Errors:**
- Ensure fair-esm is installed: `pip install fair-esm`
- Check PyTorch installation: `pip install torch`

## License

This code is provided as-is for educational and research purposes. ESM models are subject to their respective licenses from Meta AI.