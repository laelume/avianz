from setuptools import setup, find_packages

setup(
    name="avianz",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "click==8.2.1",
        "Cython==3.1.1",
        "h5py==3.13.0",
        "jsonschema==4.24.0",
        "librosa==0.11.0",
        "lxml==5.4.0",
        "openpyxl==3.1.5",
        "matplotlib==3.10.3",
        "numba==0.61.2",
        "numpy==2.1.3",
        "pyFFTW==0.15.0",
        "pyqtgraph==0.13.7",
        "resampy==0.4.3",
        "scipy==1.15.3",
        "scikit-image==0.25.2",
        "soundfile==0.13.1",
        "tensorflow==2.19.0",
        "PyQt6==6.9.0"
    ],
)