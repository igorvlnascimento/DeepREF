import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='deepref',  
        version='0.5',
        author="Igor Nascimento",
        author_email="igorvlnascimento@gmail.com",
        description="A framework for optimized deep learning-based relation classification",
        url="https://github.com/igorvlnascimento/DeepREF",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel']
     )