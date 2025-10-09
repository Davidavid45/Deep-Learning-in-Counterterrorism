#!/bin/bash

echo "================================================"
echo "CLEANING REPO FOR GIT PUSH"
echo "================================================"

# 1. Remove old Jupyter notebooks (legacy work)
echo ""
echo "[1/7] Removing legacy notebooks..."
rm -rf legacy/
echo "   ✓ Removed legacy/"

# 2. Remove pickle files (can be regenerated)
echo ""
echo "[2/7] Removing pickle files..."
find . -name "*.pkl" -delete
echo "   ✓ Removed *.pkl files"

# 3. Remove log files
echo ""
echo "[3/7] Removing log files..."
find . -name "*.log" -delete
rm -f nohup.out feature_ablation.log
echo "   ✓ Removed *.log files"

# 4. Remove Python cache
echo ""
echo "[4/7] Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
echo "   ✓ Removed __pycache__ and *.pyc"

# 5. Remove OS files
echo ""
echo "[5/7] Removing OS files..."
find . -name ".DS_Store" -delete
echo "   ✓ Removed .DS_Store files"

# 6. Clean up docs (optional - keep data_statement.md)
echo ""
echo "[6/7] Checking docs folder..."
if [ -d "docs" ]; then
    echo "   ℹ Keeping docs/data_statement.md"
else
    echo "   ℹ No docs folder"
fi

# 7. Remove old AI-generated helper files (optional)
echo ""
echo "[7/7] Removing temporary documentation..."
rm -f CLEANUP_CHECKLIST.md PAPER_ASSETS_README.md 2>/dev/null
echo "   ✓ Cleaned up temporary docs"

echo ""
echo "================================================"
echo "✅ CLEANUP COMPLETE!"
echo "================================================"
echo ""
echo "Files removed:"
echo "  - legacy/ (old notebooks)"
echo "  - *.pkl (pickle files)"
echo "  - *.log (log files)"
echo "  - __pycache__/ (Python cache)"
echo "  - .DS_Store (OS files)"
echo ""
echo "Files kept:"
echo "  - All source code (src/)"
echo "  - Results (reports/)"
echo "  - Configurations (configs/)"
echo "  - Documentation (README.md, docs/)"
echo ""
echo "Next steps:"
echo "  1. git status (verify clean)"
echo "  2. git add ."
echo "  3. git commit -m 'Clean professional codebase'"
echo "  4. git push"
echo ""
