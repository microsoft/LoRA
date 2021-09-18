import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loralib",
    version="0.1.0",
    author="Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen",
    author_email="edward.hu@microsoft.com",
    description="PyTorch implementation of low-rank adaptation (LoRA), a parameter-efficient approach to adapt a large pre-trained deep learning model which obtains performance on-par with full fine-tuning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/LoRA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)