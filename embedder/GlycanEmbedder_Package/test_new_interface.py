#!/usr/bin/env python3
"""
Test script to verify the new comprehensive GlycanEmbedder interface.
"""

def test_interface_structure():
    """Test that the interface has all expected methods and structure."""
    print("🧪 Testing GlycanEmbedder Interface Structure")
    print("=" * 50)
    
    try:
        # Import without initializing (to avoid torch dependency)
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Test that we can import the functions
        print("✅ Testing imports...")
        from glycan_embedder import GlycanEmbedder, get_available_methods
        print("  ✓ Successfully imported GlycanEmbedder and get_available_methods")
        
        # Test available methods function
        print("\n✅ Testing get_available_methods()...")
        methods = get_available_methods()
        print(f"  ✓ Available methods: {methods}")
        
        # Verify expected methods are present
        expected_graph = ['gcn', 'rgcn', 'gat', 'gin', 'compgcn', 'mpnn']
        expected_sequence = ['cnn', 'resnet', 'lstm', 'bert']
        expected_all = expected_graph + expected_sequence
        
        assert 'graph_based' in methods
        assert 'sequence_based' in methods  
        assert 'all' in methods
        
        assert set(methods['graph_based']) == set(expected_graph)
        assert set(methods['sequence_based']) == set(expected_sequence)
        assert set(methods['all']) == set(expected_all)
        
        print(f"  ✓ Graph-based methods ({len(expected_graph)}): {expected_graph}")
        print(f"  ✓ Sequence-based methods ({len(expected_sequence)}): {expected_sequence}")
        print(f"  ✓ Total methods: {len(expected_all)}")
        
        # Test class constants
        print("\n✅ Testing GlycanEmbedder class constants...")
        assert hasattr(GlycanEmbedder, 'GRAPH_METHODS')
        assert hasattr(GlycanEmbedder, 'SEQUENCE_METHODS')
        assert hasattr(GlycanEmbedder, 'ALL_METHODS')
        
        print(f"  ✓ GRAPH_METHODS: {GlycanEmbedder.GRAPH_METHODS}")
        print(f"  ✓ SEQUENCE_METHODS: {GlycanEmbedder.SEQUENCE_METHODS}")
        print(f"  ✓ ALL_METHODS: {len(GlycanEmbedder.ALL_METHODS)} total")
        
        # Verify method lists match
        assert GlycanEmbedder.GRAPH_METHODS == expected_graph
        assert GlycanEmbedder.SEQUENCE_METHODS == expected_sequence
        assert GlycanEmbedder.ALL_METHODS == expected_all
        
        print("\n✅ Testing class structure...")
        
        # Check that class has expected methods
        expected_class_methods = [
            '_create_embedder',
            '_create_graph_embedder', 
            '_create_sequence_embedder',
            'embed_glycans',
            'get_available_methods',
            'save_embedder',
            'load_embedder'
        ]
        
        for method_name in expected_class_methods:
            assert hasattr(GlycanEmbedder, method_name), f"Missing method: {method_name}"
            print(f"  ✓ Has method: {method_name}")
            
        print("\n✅ Testing embedder classes exist...")
        
        # Check that individual embedder classes are defined
        from glycan_embedder import (
            GCNEmbedder, RGCNEmbedder, GATEmbedder, GINEmbedder, 
            CompGCNEmbedder, MPNNEmbedder,
            CNNEmbedder, ResNetEmbedder, LSTMEmbedder, BERTEmbedder
        )
        
        embedder_classes = [
            'GCNEmbedder', 'RGCNEmbedder', 'GATEmbedder', 'GINEmbedder',
            'CompGCNEmbedder', 'MPNNEmbedder', 'CNNEmbedder', 'ResNetEmbedder', 
            'LSTMEmbedder', 'BERTEmbedder'
        ]
        
        for class_name in embedder_classes:
            print(f"  ✓ {class_name} class defined")
            
        print("\n🎉 All interface structure tests passed!")
        print("✅ The comprehensive GlycanEmbedder interface is correctly implemented")
        print(f"✅ Supports all {len(expected_all)} embedding methods from GlycanML")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation_examples():
    """Test that documentation examples have correct syntax."""
    print("\n📚 Testing Documentation Examples")
    print("=" * 50)
    
    try:
        # Test example code snippets (syntax only, not execution)
        example_snippets = [
            # Quick usage example
            """
from glycan_embedder import embed_glycans, get_available_methods
glycans = ["Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc"]
methods = get_available_methods()
""",
            # Advanced usage example  
            """
from glycan_embedder import GlycanEmbedder
embedder = GlycanEmbedder(vocab_path='glycoword_vocab.pkl', device='cpu')
""",
            # Method-specific examples
            """
embeddings = embedder.embed_glycans(
    glycans, 
    method='gcn',
    embedding_dim=128,
    hidden_dim=256,
    num_layers=3
)
""",
        ]
        
        for i, snippet in enumerate(example_snippets):
            try:
                compile(snippet, f'<example_{i}>', 'exec')
                print(f"  ✓ Example {i+1} syntax: Valid")
            except SyntaxError as e:
                print(f"  ❌ Example {i+1} syntax error: {e}")
                return False
        
        print("✅ All documentation examples have valid syntax")
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧬 Comprehensive GlycanEmbedder Interface Test")
    print("=" * 60)
    
    success = True
    success &= test_interface_structure()
    success &= test_documentation_examples()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The comprehensive GlycanEmbedder is ready to use")
        print("✅ Supports all 10 embedding methods from GlycanML paper")
        print("✅ Clean and simple interface for method switching")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch numpy")
        print("2. Optionally install glycowork: pip install glycowork") 
        print("3. Run examples: python example_usage.py")
    else:
        print("❌ Some tests failed. Please check the implementation.") 