from setuptools import find_packages, setup

setup(
    name="stancedetection",
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version="0.1.0",
    description="Cross-Domain Label-Adaptive Stance Detection",
    author="Checkstep Research",
    package_dir={"": "src"},
    entry_points={},
    include_package_data=True,
    license="CC BY-NC-SA 4.0",
)
