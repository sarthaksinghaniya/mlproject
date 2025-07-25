import importlib

libs = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 
    'sklearn', 'scipy', 'tensorflow', 'torch',
    'cv2', 'skimage', 'IPython', 'jupyter',
    'xgboost', 'lightgbm', 'plotly'
]

for lib in libs:
    try:
        importlib.import_module(lib)
        print(f"✅ {lib} is installed")
    except ImportError:
        print(f"❌ {lib} is NOT installed")
