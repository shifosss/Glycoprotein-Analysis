#!/usr/bin/env python3
"""
Quick test script for GlycanEmbedder functionality.
"""

def test_imports():
    """Test that imports work."""
    try:
        from glycan_embedder import embed_glycans, GlycanEmbedder
        print("✅ Imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic embedding functionality."""
    try:
        from glycan_embedder import embed_glycans
        
        # Simple test glycans
        glycans = ["Gal(b1-4)GlcNAc", "Man(a1-3)Man"]
        
        # Test graph embeddings
        graph_embeds = embed_glycans(glycans, method='graph', embedding_dim=32)
        print(f"✅ Graph embeddings: {graph_embeds.shape}")
        
        # Test sequence embeddings
        seq_embeds = embed_glycans(glycans, method='sequence', embedding_dim=32)
        print(f"✅ Sequence embeddings: {seq_embeds.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick GlycanEmbedder Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    
    if success:
        success &= test_basic_functionality()
    
    print("=" * 40)
    if success:
        print("🎉 All tests passed! GlycanEmbedder is ready to use.")
    else:
        print("❌ Some tests failed. Check dependencies and setup.") 