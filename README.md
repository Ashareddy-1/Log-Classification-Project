Log-Classification-Project

This repository provides a modular system for text classification using multiple processing strategies — including Regex, BERT, and LLMs (Large Language Models). The framework supports model training, inference, and API deployment.

📁 Project Structure

├── __pycache__/                  # Cached Python bytecode

├── models/                      # Trained model files

├── resources/                   # Static assets, config, or class mappings

├── training/                    # Training scripts and data
│
├── classify.py                  # Unified interface for running classification

├── main.py                      # Entry point to execute the entire pipeline

├── processor_bert.py            # Classification processor using BERT

├── processor_llm.py             # Processor using external LLM APIs (e.g., OpenAI)

├── processor_regex.py           # Regex-based rule processor

├── server.py                    # Backend server/API for deployment

├── requirements.txt             # Python dependencies

├── README.md                    # Project documentation

⚙️ Features

🔄 Pluggable processing modules: Choose between regex, BERT, or LLMs.

🧪 Training pipeline: Custom training workflows in the training/ folder.

🌐 API endpoint: Easily deploy the model using server.py.

🧩 Classifier abstraction: Central control via classify.py.

🚀 Getting Started

1. Clone the Repository

2. Install Dependencies
   
pip install -r requirements.txt

▶️ Usage

Run Classification via CLI

python classify.py --text "Enter your sentence here" --processor bert

Available processor options:

regex

bert

llm

Run API Server

python server.py

Run Full Pipeline

python main.py

🔐 LLM API Configuration

If you're using an LLM (like OpenAI), set your API key:

export OPENAI_API_KEY=your_key_here

📂 Key Files Explained

File/Folder	Description

processor_regex.py	Rule-based classification using regular expressions

processor_bert.py	Transformer-based classification with BERT

processor_llm.py	Utilizes external LLM APIs for classification

classify.py	Command-line interface for classification

main.py	Orchestrates the full pipeline

server.py	Deploys the classifier as a web API

training/	Scripts and utilities for training new models

models/	Trained model files and weights
