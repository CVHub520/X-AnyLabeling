# ===== Build packages =====
# For CPU
sed -i'' -e 's/\_\_preferred_device\_\_[ ]*=[ ]*\"[A-Za-z0-9]*\"/__preferred_device__ = "CPU"/g' anylabeling/app_info.py
python -m build --no-isolation --outdir wheels_dist
# For GPU
sed -i'' -e 's/\_\_preferred_device\_\_[ ]*=[ ]*\"[A-Za-z0-9]*\"/__preferred_device__ = "GPU"/g' anylabeling/app_info.py
python -m build --no-isolation --outdir wheels_dist
# Restore to CPU (default option)
sed -i'' -e 's/\_\_preferred_device\_\_[ ]*=[ ]*\"[A-Za-z0-9]*\"/__preferred_device__ = "CPU"/g' anylabeling/app_info.py

# ===== Publish to PyPi =====
twine upload --skip-existing wheels_dist/*
