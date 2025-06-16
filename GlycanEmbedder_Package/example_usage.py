#!/usr/bin/env python3
"""
Comprehensive Examples for Refactored GlycanEmbedder Package

This file demonstrates all 10 embedding methods from the GlycanML paper
and shows how to switch between different embedders efficiently.
"""

import numpy as np
import torch
from glycan_embedder import GlycanEmbedder, embed_glycans, get_available_methods
from glycan_embedder import GlycanGCN, GlycanLSTM, GlycanBERT  # Direct access to embedder classes
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

print("🧬 GlycanEmbedder: Comprehensive Usage Examples")
print("=" * 60)

vocab = 'glycoword_vocab.pkl'


def example_quick_usage():
    """Demonstrate quick usage with different methods."""
    print("\n📖 Example 1: Quick Usage with Different Methods")
    print("-" * 50)

    # Sample glycans
    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)Glc",
        "GlcNAc(b1-4)Man(a1-3)[GlcNAc(b1-4)Man(a1-6)]Man(b1-4)GlcNAc"
    ]

    print("Available embedding methods:")
    methods = get_available_methods()
    for category, method_list in methods.items():
        print(f"  {category}: {method_list}")

    print(f"\nTesting with {len(glycans)} glycans...")

    # Test all methods
    embedding_dim = 64
    for method in ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn', 'cnn', 'resnet', 'lstm', 'bert']:
        embeddings = embed_glycans(glycans, method=method, embedding_dim=embedding_dim, 
                                   vocab_path=vocab, device='cuda')
        print(f"  {method.upper():>8}: {embeddings.shape}")


def example_advanced_usage():
    """Demonstrate advanced usage with the GlycanEmbedder class."""
    print("\n🔧 Example 2: Advanced Usage with GlycanEmbedder Class")
    print("-" * 50)

    # Initialize embedder
    embedder = GlycanEmbedder(vocab_path='glycoword_vocab.pkl', device='cpu')

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)Glc"
    ]

    print("Testing different embedding methods:")

    # Graph-based methods with custom parameters
    print("\n  Graph-based methods:")

    # GCN with custom hidden dimensions
    gcn_embeds = embedder.embed_glycans(
        glycans,
        method='gcn',
        embedding_dim=128,
        input_dim=256,
        hidden_dims=[256, 256, 128],
        short_cut=True,
        concat_hidden=False,
        readout="dual"
    )
    print(f"    GCN (custom): {gcn_embeds.shape}")

    # RGCN with batch normalization
    rgcn_embeds = embedder.embed_glycans(
        glycans,
        method='rgcn',
        embedding_dim=128,
        hidden_dims=[256, 128],
        batch_norm=True
    )
    print(f"    RGCN: {rgcn_embeds.shape}")

    # GAT with 8 attention heads
    gat_embeds = embedder.embed_glycans(
        glycans,
        method='gat',
        embedding_dim=128,
        num_heads=8,
        hidden_dims=[256, 128]
    )
    print(f"    GAT (8 heads): {gat_embeds.shape}")

    # GIN with custom epsilon
    gin_embeds = embedder.embed_glycans(
        glycans,
        method='gin',
        embedding_dim=128,
        eps=0.1,
        learn_eps=True,
        num_mlp_layer=3
    )
    print(f"    GIN (eps=0.1): {gin_embeds.shape}")

    # CompGCN with custom composition
    compgcn_embeds = embedder.embed_glycans(
        glycans,
        method='compgcn',
        embedding_dim=128,
        composition="multiply"
    )
    print(f"    CompGCN: {compgcn_embeds.shape}")

    # MPNN with custom message passing steps
    mpnn_embeds = embedder.embed_glycans(
        glycans,
        method='mpnn',
        embedding_dim=128,
        num_layer=5,
        num_gru_layer=2
    )
    print(f"    MPNN (5 layers): {mpnn_embeds.shape}")

    # Sequence-based methods with custom parameters
    print("\n  Sequence-based methods:")

    # CNN with custom kernel size
    cnn_embeds = embedder.embed_glycans(
        glycans,
        method='cnn',
        embedding_dim=128,
        kernel_size=5,
        hidden_dims=[512, 256, 128],
        concat_hidden=False
    )
    print(f"    CNN (k=5, concat): {cnn_embeds.shape}")

    # ResNet with custom blocks
    resnet_embeds = embedder.embed_glycans(
        glycans,
        method='resnet',
        embedding_dim=128,
        num_blocks=4,
        layer_norm=True,
        dropout=0.2
    )
    print(f"    ResNet (4 blocks): {resnet_embeds.shape}")

    # LSTM with custom layers
    lstm_embeds = embedder.embed_glycans(
        glycans,
        method='lstm',
        embedding_dim=128,
        num_layers=3,
        bidirectional=True,
        dropout=0.1
    )
    print(f"    LSTM (3 layers, bidir): {lstm_embeds.shape}")

    # BERT with custom architecture
    bert_embeds = embedder.embed_glycans(
        glycans,
        method='bert',
        embedding_dim=768,
        num_layers=8,
        num_heads=12,
        intermediate_dim=3072
    )
    print(f"    BERT (8L, 12H): {bert_embeds.shape}")


def example_direct_embedder_usage():
    """Demonstrate using embedder classes directly (like in graph_models.py)."""
    print("\n🎯 Example 3: Direct Embedder Class Usage")
    print("-" * 50)

    # Load vocabulary
    embedder = GlycanEmbedder(vocab_path=vocab, device='cuda')

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    ]

    # Parse glycans
    unit_ids_batch = []
    glycoword_ids_batch = []
    for glycan in glycans:
        unit_ids, glycoword_ids = embedder._parse_glycan_to_units_and_links(glycan)
        unit_ids_batch.append(unit_ids)
        glycoword_ids_batch.append(glycoword_ids)

    # Create graph object
    graph = embedder._create_graph_object(unit_ids_batch, glycoword_ids_batch, 'gcn')

    # Use GCN embedder directly
    print("Using GlycanGCN directly:")
    gcn_model = GlycanGCN(
        input_dim=128,
        hidden_dims=[256, 256, 128],
        num_unit=len(embedder.units),
        short_cut=True,
        batch_norm=True,
        activation="relu",
        concat_hidden=False,
        readout="dual"
    ).to('cuda')

    with torch.no_grad():
        output = gcn_model(graph)
        print(f"  Graph features: {output['graph_feature'].shape}")
        print(f"  Node features: {output['node_feature'].shape}")
        print(f"  Output dimension: {gcn_model.get_output_dim()}")

    # Use LSTM embedder directly
    print("\nUsing GlycanLSTM directly:")
    graph_seq = embedder._create_graph_object(unit_ids_batch, glycoword_ids_batch, 'lstm')

    lstm_model = GlycanLSTM(
        input_dim=1024,
        hidden_dim=512,
        glycoword_dim=len(embedder.glycowords),
        num_layers=2,
        bidirectional=True,
        layer_norm=True
    ).to('cuda')

    with torch.no_grad():
        output = lstm_model(graph_seq)
        print(f"  Graph features: {output['graph_feature'].shape}")
        print(f"  Glycoword features: {output['glycoword_feature'].shape}")
        print(f"  Output dimension: {lstm_model.get_output_dim()}")


def example_method_comparison():
    """Compare different embedding methods on the same glycans."""
    print("\n📊 Example 4: Method Comparison Analysis")
    print("-" * 50)

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",  # Blood group A antigen
        "Gal(a1-4)[Fuc(a1-3)]GlcNAc(b1-3)Gal(b1-4)Glc",  # Lewis X antigen
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",  # N-glycan core
        "Neu5Ac(a2-3)Gal(b1-4)Glc",  # Simple sialylated structure
    ]

    # Compare graph-based vs sequence-based approaches
    methods_to_compare = ['gcn', 'gat', 'cnn', 'lstm', 'bert']
    embedding_dim = 128

    print("Computing embeddings with different methods...")
    embeddings_dict = {}

    for method in methods_to_compare:
        embeddings = embed_glycans(glycans, method=method, embedding_dim=embedding_dim, vocab_path=vocab, device='cuda')
        embeddings_dict[method] = embeddings
        print(f"  {method.upper()}: ✓")

    # Compute pairwise similarities within each method
    print("\nPairwise cosine similarities:")
    for method, embeddings in embeddings_dict.items():
        similarities = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        print(f"  {method.upper()}: avg similarity = {avg_similarity:.3f}")

    # Cross-method comparison
    print("\nCross-method embedding similarities:")
    gcn_embeds = embeddings_dict['gcn']
    for method, embeddings in embeddings_dict.items():
        if method != 'gcn':
            # Compare first glycan embeddings across methods
            sim = cosine_similarity([gcn_embeds[0]], [embeddings[0]])[0, 0]
            print(f"  GCN vs {method.upper()}: {sim:.3f}")


def example_readout_comparison():
    """Demonstrate different readout functions."""
    print("\n🔍 Example 5: Readout Function Comparison")
    print("-" * 50)

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
        "Neu5Ac(a2-3)Gal(b1-4)Glc"
    ]

    readout_functions = ['sum', 'mean', 'max', 'attention', 'dual']
    embedding_dim = 128

    print("Testing different readout functions with GCN:")
    embedder = GlycanEmbedder(vocab_path=vocab, device='cuda')

    for readout in readout_functions:
        if readout == 'attention':
            # Attention readout needs proper dimensions
            embeddings = embedder.embed_glycans(
                glycans,
                method='gcn',
                embedding_dim=embedding_dim,
                readout=readout,
                concat_hidden=False  # Important for attention
            )
        else:
            embeddings = embedder.embed_glycans(
                glycans,
                method='gcn',
                embedding_dim=embedding_dim,
                readout=readout
            )
        print(f"  {readout:>10}: shape = {embeddings.shape}")


def example_machine_learning_integration():
    """Demonstrate integration with machine learning workflows."""
    print("\n🤖 Example 6: Machine Learning Integration")
    print("-" * 50)

    # Generate synthetic dataset
    np.random.seed(42)

    # Create glycans with different structural patterns
    n_linked_glycans = [
                           "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
                           "GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
                           "GalNAc(b1-4)GlcNAc(b1-2)Man(a1-3)[GalNAc(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
                       ] * 5

    o_linked_glycans = [
                           "GalNAc(a1-3)Gal",
                           "Neu5Ac(a2-3)Gal(b1-3)GalNAc",
                           "GlcNAc(b1-3)Gal(b1-3)GalNAc"
                       ] * 5

    free_glycans = [
                       "Glc(a1-4)Glc",
                       "Fru(b2-1)Glc",
                       "Gal(a1-6)Glc"
                   ] * 5

    all_glycans = n_linked_glycans + o_linked_glycans + free_glycans
    labels = [0] * len(n_linked_glycans) + [1] * len(o_linked_glycans) + [2] * len(free_glycans)

    print(f"Dataset: {len(all_glycans)} glycans, 3 classes")

    # Test different embedding methods for classification
    methods = ['gcn', 'rgcn', 'cnn', 'lstm']

    for method in methods:
        print(f"\n  Testing {method.upper()} for classification:")

        # Get embeddings with proper configuration
        if method in ['gcn', 'rgcn']:
            embeddings = embed_glycans(
                all_glycans,
                method=method,
                embedding_dim=128,
                hidden_dims=[256, 128],
                readout='dual',
                vocab_path=vocab,
                device='cuda'
            )
        else:
            embeddings = embed_glycans(
                all_glycans,
                method=method,
                embedding_dim=128,
                vocab_path=vocab,
                device='cuda'
            )

        # Simple train/test split
        train_size = int(0.7 * len(embeddings))
        indices = np.random.permutation(len(embeddings))

        X_train = embeddings[indices[:train_size]]
        X_test = embeddings[indices[train_size:]]
        y_train = np.array(labels)[indices[:train_size]]
        y_test = np.array(labels)[indices[train_size:]]

        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"    Train accuracy: {train_acc:.3f}")
        print(f"    Test accuracy: {test_acc:.3f}")

        # Clustering analysis
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate clustering purity
        from collections import Counter
        cluster_purities = []
        for cluster_id in range(3):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_true_labels = np.array(labels)[cluster_mask]
                most_common = Counter(cluster_true_labels).most_common(1)[0][1]
                purity = most_common / len(cluster_true_labels)
                cluster_purities.append(purity)

        avg_purity = np.mean(cluster_purities) if cluster_purities else 0
        print(f"    Clustering purity: {avg_purity:.3f}")


def example_save_load_embedders():
    """Demonstrate saving and loading trained embedders."""
    print("\n💾 Example 7: Saving and Loading Embedders")
    print("-" * 50)

    # Create embedder
    embedder = GlycanEmbedder(vocab_path=vocab, device='cpu')

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    ]

    print("Creating embedders with specific architectures...")

    # Create GCN embedder with specific architecture
    gcn_embeds = embedder.embed_glycans(
        glycans,
        method='gcn',
        embedding_dim=128,
        hidden_dims=[256, 256, 128],
        readout='dual'
    )
    print(f"  GCN embeddings: {gcn_embeds.shape}")

    # Create LSTM embedder
    lstm_embeds = embedder.embed_glycans(
        glycans,
        method='lstm',
        embedding_dim=128,
        num_layers=3
    )
    print(f"  LSTM embeddings: {lstm_embeds.shape}")

    # Access the embedder models directly
    gcn_key = "gcn_128_128"  # method_inputdim_embeddingdim
    lstm_key = "lstm_1024_128"

    if gcn_key in embedder._embedders:
        gcn_model = embedder._embedders[gcn_key]
        print(f"\n  GCN model architecture:")
        print(f"    Input dim: {gcn_model.input_dim}")
        print(f"    Output dim: {gcn_model.get_output_dim()}")
        print(f"    Layers: {len(gcn_model.layers)}")

        # Save state dict
        torch.save(gcn_model.state_dict(), 'gcn_embedder.pth')
        print("  ✓ GCN embedder saved")

    if lstm_key in embedder._embedders:
        lstm_model = embedder._embedders[lstm_key]
        print(f"\n  LSTM model architecture:")
        print(f"    Hidden dim: {lstm_model.hidden_dim}")
        print(f"    Num layers: {lstm_model.num_layers}")

        # Save state dict
        torch.save(lstm_model.state_dict(), 'lstm_embedder.pth')
        print("  ✓ LSTM embedder saved")

    # Create new embedder and load weights
    print("\n  Loading embedders in new session...")
    new_embedder = GlycanEmbedder(vocab_path=vocab, device='cpu')

    # Recreate models with same architecture
    _ = new_embedder.embed_glycans(
        glycans[:1],  # Just one to create the model
        method='gcn',
        embedding_dim=128,
        hidden_dims=[256, 256, 128],
        readout='dual'
    )

    _ = new_embedder.embed_glycans(
        glycans[:1],
        method='lstm',
        embedding_dim=128,
        num_layers=3
    )

    # Load weights
    if gcn_key in new_embedder._embedders:
        new_embedder._embedders[gcn_key].load_state_dict(torch.load('gcn_embedder.pth'))
        print("  ✓ GCN embedder loaded")

    if lstm_key in new_embedder._embedders:
        new_embedder._embedders[lstm_key].load_state_dict(torch.load('lstm_embedder.pth'))
        print("  ✓ LSTM embedder loaded")

    # Test loaded embedders
    new_gcn_embeds = new_embedder.embed_glycans(
        glycans,
        method='gcn',
        embedding_dim=128,
        hidden_dims=[256, 256, 128],
        readout='dual'
    )
    new_lstm_embeds = new_embedder.embed_glycans(
        glycans,
        method='lstm',
        embedding_dim=128,
        num_layers=3
    )

    # Check if embeddings are the same
    gcn_diff = torch.mean(torch.abs(gcn_embeds - new_gcn_embeds)).item()
    lstm_diff = torch.mean(torch.abs(lstm_embeds - new_lstm_embeds)).item()

    print(f"\n  GCN embedding difference: {gcn_diff:.6f}")
    print(f"  LSTM embedding difference: {lstm_diff:.6f}")


def example_gpu_usage():
    """Demonstrate GPU usage if available."""
    print("\n🚀 Example 8: GPU Usage (if available)")
    print("-" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create embedder with GPU if available
    embedder = GlycanEmbedder(vocab_path=vocab, device=device)

    glycans = [
                  "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
                  "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc",
                  "Neu5Ac(a2-3)Gal(b1-4)Glc"
              ] * 10  # More glycans for GPU benefit

    print(f"\nProcessing {len(glycans)} glycans...")

    # Test different methods on GPU
    methods = ['gcn', 'gat', 'cnn', 'bert']
    for method in methods:
        if method == 'bert':
            embeddings = embedder.embed_glycans(
                glycans,
                method=method,
                embedding_dim=256,
                hidden_dim=512
            )
        else:
            embeddings = embedder.embed_glycans(
                glycans,
                method=method,
                embedding_dim=256,
                hidden_dims=[512, 256] if method in ['gcn', 'gat'] else [512, 256]
            )
        print(f"  {method.upper()}: {embeddings.shape} on {embeddings.device}")


def example_custom_architectures():
    """Demonstrate custom architectures for different use cases."""
    print("\n🏗️ Example 9: Custom Architectures")
    print("-" * 50)

    glycans = [
        "Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc",
        "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    ]

    embedder = GlycanEmbedder(vocab_path=vocab, device='cuda')

    print("1. Deep GCN for complex structural patterns:")
    deep_gcn_embeds = embedder.embed_glycans(
        glycans,
        method='gcn',
        embedding_dim=64,
        input_dim=512,
        hidden_dims=[512, 256, 128, 64],  # 4 layers
        short_cut=True,
        batch_norm=True,
        activation='gelu',
        readout='attention'
    )
    print(f"   Shape: {deep_gcn_embeds.shape}")

    print("\n2. Wide BERT for sequence modeling:")
    wide_bert_embeds = embedder.embed_glycans(
        glycans,
        method='bert',
        embedding_dim=1024,
        num_layers=12,
        num_heads=16,
        intermediate_dim=4096,
        hidden_dropout=0.1,
        attention_dropout=0.1
    )
    print(f"   Shape: {wide_bert_embeds.shape}")

    print("\n3. Hybrid approach - concatenate multiple methods:")
    gcn_emb = embedder.embed_glycans(glycans, method='gcn', embedding_dim=128)
    lstm_emb = embedder.embed_glycans(glycans, method='lstm', embedding_dim=128)
    bert_emb = embedder.embed_glycans(glycans, method='bert', embedding_dim=128)

    hybrid_emb = np.concatenate([gcn_emb.cpu(), lstm_emb.cpu(), bert_emb.cpu()], axis=1)
    print(f"   Hybrid shape: {hybrid_emb.shape} (captures both structure and sequence)")


if __name__ == "__main__":
    # Run all examples
    example_quick_usage()
    example_advanced_usage()
    example_direct_embedder_usage()
    example_method_comparison()
    example_readout_comparison()
    example_machine_learning_integration()
    example_save_load_embedders()
    example_gpu_usage()
    example_custom_architectures()

    print("\n" + "=" * 60)
    print("🎉 All examples completed successfully!")
    print("The refactored embedder provides:")
    print("  ✓ Cleaner architecture matching graph_models.py patterns")
    print("  ✓ Direct access to embedder classes")
    print("  ✓ Consistent forward() interface")
    print("  ✓ Flexible customization options")
    print("  ✓ All 10 embedding methods fully functional")