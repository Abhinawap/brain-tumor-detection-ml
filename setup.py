from setuptools import setup, find_packages

setup(
    name="brain_tumor_segmentation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.0",
    ],
    python_requires=">=3.11",
)
