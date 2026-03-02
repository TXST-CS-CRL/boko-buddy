#!/bin/bash

echo "🚀 Starting TXST AI Tutor Environment Setup..."

# 1. Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate Environment
source venv/bin/activate

# 3. Upgrade Pip
pip install --upgrade pip

# 4. Install Core Libraries
echo "📦 Installing AI Frameworks..."
pip install llama-index llama-index-core llama-index-llms-openai \
            llama-index-embeddings-openai llama-index-readers-file

# 5. Install UI & File Parsers
echo "📦 Installing UI and Document Parsers..."
pip install streamlit python-pptx python-dotenv

echo "✅ Environment is ready!"
echo "------------------------------------------------"
echo "Next Steps:"
echo "0. source venv/bin/activate" 
echo "1. export OPENAI_API_KEY='your_key'"
echo "2. python3 ingest.py [folder_path]"
echo "3. streamlit run app.py"
