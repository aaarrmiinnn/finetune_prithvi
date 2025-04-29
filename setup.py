from setuptools import setup, find_packages

setup(
    name="finetune_prithvi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.0.0",
        "tensorboard>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "netCDF4>=1.5.0",
        "xarray>=0.16.0",
        "rioxarray>=0.13.0",
        "rasterio>=1.2.0",
        "gdal>=3.0.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
        "pyyaml>=5.4.0",
        "wandb>=0.12.0",
    ],
) 