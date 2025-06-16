@echo off
echo 🚀 Installing GlycanEmbedder dependencies...

rem Install basic requirements
pip install torch numpy

rem Try to install optional dependencies
echo 📦 Installing optional dependencies...
pip install glycowork scikit-learn

echo ✅ Installation complete!
echo.
echo 📋 Quick test:
python -c "from glycan_embedder import embed_glycans; print('✅ Import successful!')"
echo.
echo 📖 See README_GlycanEmbedder.md for usage examples
pause 