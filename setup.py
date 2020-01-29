import setuptools

setuptools.setup(
    name="fairensics",
    version="0.0.1",
    author="Niki Kilbertus",
    author_email="nk470@cam.ac.uk",
    description="A library for fair machine learning.",
    url="https://github.com/nikikilbertus/fairensics",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "cvxpy",
        "dccp",
        "scipy",
        "sklearn",
        "aif360",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
    ],
    python_requires=">=3.6",
)
