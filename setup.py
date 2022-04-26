import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='deepref',  
        version='0.1',
        author="Tianyu Gao",
        author_email="igorvlnascimento@gmail.com",
        description="An open source toolkit for relation extraction",
        url="https://github.com/igorvlnascimento/DeepREF",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel']
     )