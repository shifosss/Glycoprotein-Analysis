# Glycan-Protein Pair Embedder

A clean and efficient embedder for combining glycan and protein sequence representations for machine learning models.

## Features

- 🧬 **Automatic Dimension Matching**: Glycan embeddings automatically match protein dimensions
- 🔀 **Two Fusion Methods**: Simple concatenation and sophisticated attention-based fusion
- 📊 **Batch Processing**: Efficient handling of large datasets
- 🎯 **ML-Ready Output**: Returns numpy arrays or PyTorch tensors ready for model training
- 🚀 **Simple & Clean**: Minimal code with maximum functionality

## Installation

```bash
# Required dependencies
pip install torch numpy

# For protein embedding
pip install fair-esm

# For glycan embedding (optional but recommended)
pip install glycowork
```

## Quick Start

```python
from glycan_protein_embedder import embed_glycan_protein_pairs

# Define glycan-protein pairs
pairs = [
    ("Gal(a1-3)Gal(b1-4)GlcNAc", "MKTVRQERLKSIVRILERSKEPVSGAQ..."),
    ("Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc", "KALTARQQEVFDLIRDHISQTGMP..."),
]

# Get embeddings with concatenation fusion
embeddings = embed_glycan_protein_pairs(
    pairs,
    protein_model="650M",      # ESM2 model size
    glycan_method="lstm",      # Glycan embedding method
    fusion_method="concat"     # Fusion method
)
print(f"Shape: {embeddings.shape}")  # (2, 2560) for 650M model
```

## Detailed Usage

### Basic Embedding

```python
from glycan_protein_embedder import GlycanProteinPairEmbedder

# Create embedder
embedder = GlycanProteinPairEmbedder(
    protein_model="650M",  # or "3B"
    protein_model_dir="resources/esm-model-weights",
    glycan_method="lstm",  # or gcn, bert, etc.
    glycan_vocab_path="path/to/vocab.pkl",  # optional
    fusion_method="concat",  # or "attention"
    device="cuda"  # or "cpu"
)

# Embed pairs
embeddings = embedder.embed_pairs(pairs, batch_size=32)
print(f"Output dimension: {embedder.get_output_dim()}")
```

### Fusion Methods

#### 1. Concatenation Fusion (Default)
Simple concatenation of glycan and protein embeddings:

```python
embedder = GlycanProteinPairEmbedder(fusion_method="concat")
# Output: [glycan_embedding || protein_embedding]
# Dimension: 2 × protein_embedding_dim
```

#### 2. Attention-Based Fusion
Sophisticated fusion using cross-attention mechanism:

```python
embedder = GlycanProteinPairEmbedder(fusion_method="attention")
# Output: Attention-weighted combination
# Dimension: protein_embedding_dim
```

### Output Formats

```python
# Get numpy array (default) - best for scikit-learn, etc.
embeddings_np = embedder.embed_pairs(pairs, return_numpy=True)

# Get PyTorch tensor - best for PyTorch models
embeddings_torch = embedder.embed_pairs(pairs, return_numpy=False)
```

## Integration with ML Models

### PyTorch Example

```python
import torch
import torch.nn as nn

class GlycanProteinPredictor(nn.Module):
    def __init__(self, embedder, num_classes=2):
        super().__init__()
        self.embedder = embedder
        
        # Build prediction head based on embedder output dimension
        self.predictor = nn.Sequential(
            nn.Linear(embedder.get_output_dim(), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pairs):
        # Get embeddings (no gradient needed for embedders)
        with torch.no_grad():
            embeddings = self.embedder.embed_pairs(pairs, return_numpy=False)
        
        # Predict
        return self.predictor(embeddings)

# Usage
model = GlycanProteinPredictor(embedder, num_classes=2)
logits = model(pairs)
```

### Scikit-learn Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Get embeddings
X = embedder.embed_pairs(pairs)  # numpy array
y = labels  # your labels

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

## Available Options

### Protein Models
- `650M`: Faster, less memory, dimension 1280
- `3B`: Better representations, more memory, dimension 2560

### Glycan Methods
**Graph-based**: `gcn`, `rgcn`, `gat`, `gin`, `compgcn`, `mpnn`  
**Sequence-based**: `cnn`, `resnet`, `lstm`, `bert`

### Output Dimensions
| Protein Model | Fusion Method | Output Dimension |
|--------------|---------------|------------------|
| 650M | concat | 2560 |
| 650M | attention | 1280 |
| 3B | concat | 5120 |
| 3B | attention | 2560 |

## Batch Processing

For large datasets, the embedder automatically processes in batches:

```python
# Process 10,000 pairs efficiently
large_pairs = [...]  # Your large dataset
embeddings = embedder.embed_pairs(large_pairs, batch_size=64)
```

## Saving and Loading

For attention fusion, you can save the learned attention weights:

```python
# Save attention weights
embedder.save_attention_weights("attention_weights.pt")

# Load in new session
new_embedder = GlycanProteinPairEmbedder(fusion_method="attention")
new_embedder.load_attention_weights("attention_weights.pt")
```

## Best Practices

1. **Model Selection**:
   - Use `650M` for faster processing and when memory is limited
   - Use `3B` for best representations when resources allow

2. **Fusion Method**:
   - Use `concat` for simplicity and when you have enough data
   - Use `attention` for smaller datasets or when interpretability matters

3. **Glycan Methods**:
   - `lstm` or `bert` work well for most cases
   - `gcn` or `gat` if structural properties are important

4. **Batch Size**:
   - Larger batches (64-128) for GPU
   - Smaller batches (16-32) for CPU or limited memory

## Example: Complete Training Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class GlycanProteinDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

# Initialize embedder
embedder = GlycanProteinPairEmbedder(
    protein_model="650M",
    glycan_method="lstm",
    fusion_method="attention",
    device="cuda"
)

# Create model
model = GlycanProteinPredictor(embedder, num_classes=2).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
dataset = GlycanProteinDataset(pairs, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_pairs, batch_labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch_pairs)
        loss = criterion(logits, batch_labels.to("cuda"))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Troubleshooting

**Out of Memory**:
- Use smaller batch sizes
- Use `650M` instead of `3B` model
- Process on CPU if GPU memory is insufficient

**Slow Processing**:
- Ensure CUDA is available for GPU acceleration
- Increase batch size if memory allows
- Use `650M` model for faster processing

**Import Errors**:
- Ensure all dependencies are installed
- Check that glycan vocabulary file path is correct