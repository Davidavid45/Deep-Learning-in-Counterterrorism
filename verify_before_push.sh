#!/bin/bash

echo "═══════════════════════════════════════"
echo "PRE-PUSH VERIFICATION"
echo "═══════════════════════════════════════"

# Check 1: No large CSV files
echo ""
echo "[1/5] Checking for large CSV files in data/raw/..."
if find data/raw -name "*.csv" 2>/dev/null | grep -q .; then
    echo "   ❌ FOUND CSV files in data/raw/ - these should NOT be pushed!"
    find data/raw -name "*.csv" -exec ls -lh {} \;
else
    echo "   ✅ No CSV files in data/raw/"
fi

# Check 2: No pickle files
echo ""
echo "[2/5] Checking for pickle files..."
if find . -name "*.pkl" 2>/dev/null | grep -q .; then
    echo "   ❌ FOUND .pkl files - these should be removed!"
    find . -name "*.pkl"
else
    echo "   ✅ No pickle files"
fi

# Check 3: No log files
echo ""
echo "[3/5] Checking for log files..."
if find . -name "*.log" -o -name "nohup.out" 2>/dev/null | grep -q .; then
    echo "   ❌ FOUND log files - these should be removed!"
    find . -name "*.log" -o -name "nohup.out"
else
    echo "   ✅ No log files"
fi

# Check 4: Check total size
echo ""
echo "[4/5] Checking repository size..."
SIZE=$(du -sh . | cut -f1)
echo "   Total size: $SIZE"
if [[ "$SIZE" =~ ^[0-9]+\.?[0-9]*M$ ]]; then
    echo "   ✅ Size is good!"
elif [[ "$SIZE" =~ ^[0-9]+\.?[0-9]*G$ ]]; then
    echo "   ⚠️  WARNING: Size is in GB - might be too large!"
else
    echo "   ✅ Size looks acceptable"
fi

# Check 5: Essential files exist
echo ""
echo "[5/5] Checking essential files..."
MISSING=0
for file in README.md .gitignore Makefile requirements.txt; do
    if [ -f "$file" ]; then
        echo "   ✅ $file exists"
    else
        echo "   ❌ $file MISSING!"
        MISSING=1
    fi
done

echo ""
echo "═══════════════════════════════════════"
if [ $MISSING -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED!"
    echo "═══════════════════════════════════════"
    echo ""
    echo "Ready to push! Run:"
    echo "  git init"
    echo "  git add ."
    echo "  git commit -m 'Complete project'"
    echo "  git push -f origin main"
else
    echo "⚠️  SOME CHECKS FAILED"
    echo "═══════════════════════════════════════"
    echo "Fix the issues above before pushing"
fi
echo ""
