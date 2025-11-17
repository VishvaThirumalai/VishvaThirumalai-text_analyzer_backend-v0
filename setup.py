from setuptools import setup, find_packages

setup(
    name="text-analyzer-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-dotenv==1.0.0",
        "openai==1.3.0",
        "transformers==4.35.0",
        "torch==2.1.1",
        "pydantic==2.4.2",
        "python-multipart==0.0.6",
        "nltk==3.8.1",
        "rake-nltk==1.0.6",
        "requests==2.31.0",
    ],
    python_requires=">=3.8",
)