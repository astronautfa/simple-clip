# Import required setup tools
from setuptools import setup, find_packages  # Tools for package setup
import os  # For file and path operations


def read_long_description():
    """
    Reads the contents of README.md to use as the long description for the package.

    This function:
    1. Gets the absolute path of the current directory
    2. Locates and reads the README.md file
    3. Returns its contents as a string

    Returns:
        str: Content of README.md file
    """
    # Get the absolute path of the directory containing this setup.py file
    root = os.path.abspath(os.path.dirname(__file__))
    # Create path to README.md
    path = os.path.join(root, "README.md")
    # Open and read the file
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


# Main setup configuration
setup(
    # Basic package information
    name='simple-clip',  # Name of the package on PyPI
    # Automatically find all packages to include
    packages=find_packages(exclude=['notebooks']),
    # Excludes 'notebooks' directory
    version='0.2.0',  # Package version number
    license='MIT',  # License type

    # Package description
    description='A minimal, but effective implementation of CLIP (Contrastive Language-Image Pretraining) in PyTorch',
    long_description=read_long_description(),  # Detailed description from README.md
    long_description_content_type='text/markdown',  # Format of the long description

    # Author information
    author='Filip Basara',
    author_email='basarafilip@gmail.com',

    # Project home page
    url='https://github.com/filipbasara0/simple-clip',

    # Search keywords for PyPI
    keywords=[
        'machine learning',
        'pytorch',
        'self-supervised learning',
        'representation learning',
        'contrastive learning'
    ],

    # Package dependencies
    # Each line specifies a required package and its minimum version
    install_requires=[
        'torch>=2.1',           # PyTorch for deep learning
        'torchvision>=0.16',    # Computer vision tools for PyTorch
        'transformers>=4.35',   # Hugging Face transformers library
        'datasets>=2.15',       # Dataset handling
        'tqdm>=4.66',          # Progress bars
        'torchinfo>=1.8.0',    # Model information display
        'webdataset>=0.2.77',  # Large-scale dataset handling
        'scikit-learn>=1.3.2'  # Machine learning utilities
    ],

    # Package classifiers for PyPI
    # These help users find your package and understand its status
    classifiers=[
        'Development Status :: 4 - Beta',  # Package is in beta stage
        'Intended Audience :: Developers',  # Target audience
        'Topic :: Scientific/Engineering :: Artificial Intelligence',  # Topic category
        'License :: OSI Approved :: MIT License',  # License type
        'Programming Language :: Python :: 3.10',  # Python version support
    ],

    # Scripts to be installed
    # These files will be installed as executable scripts
    scripts=['run_training.py', 'train.py'],

    # Console script entry points
    # These create command-line commands that users can run
    entry_points={
        "console_scripts": [
            # Creates a 'train_clip' command that runs main() from run_training.py
            "train_clip = run_training:main"
        ],
    },
)
