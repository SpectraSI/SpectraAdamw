from setuptools import setup, find_packages

setup(
    name="spectra-torch",
    version="0.4.0",
    description="Frequency-Domain Gradient Optimizer for Deep LLM Architectures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Spectra Labs",
    author_email="kanishk@spectralabs.si",
    url="https://github.com/SpectraSI/SpectraAdamw",
    py_modules=["spectra"],
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
)
