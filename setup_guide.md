
## Boko Buddy - TXST AI Tutor Setup Guide (macOS only)

This guide will help you set up your personal AI tutor. We will use **LlamaIndex** to process your
course materials and **Streamlit** to provide a chat interface.


---

## Phase 1: Preparation

Before we begin, ensure you have your course materials (PDFs, PPTXs) downloaded from Canvas into a
single folder (e.g., `~/Desktop/Course_Material`).


### 1. Install System Tools

Open **Terminal** (Cmd + Space, type "Terminal") and paste these commands one by one. You can skip
this step if you already have `git` and `git-lfs` installed on your system. 

* **Homebrew**:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```


* **Git & Git-LFS**:
```bash
brew install git git-lfs

```

### 2. Get the Code (Clone Repo)

Now, download the project files to your computer:

```bash
git clone https://github.com/TXST-CS-CRL/boko-buddy
cd YOUR_REPO_NAME

```

### 3. Initialize Git-LFS (Optional but Recommended)

If you plan to sync large vector databases later:

```bash
git lfs install

```

### Automated Setup of Python Environment and AI Libraries

To save time, you can skip steps 4 and 5 and run the automated `setup.sh` script that handles the
Python environment and libraries. 

```bash
bash setup.sh

```

### 4. Set Up Python Environment

We recommend using a virtual environment to keep your machine clean:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade the installer
pip install --upgrade pip

```

### 5. Install AI Libraries

Paste this block to install the "brain" and the "interface":

```bash
pip install llama-index llama-index-core llama-index-llms-openai \
            llama-index-embeddings-openai llama-index-readers-file \
            python-pptx streamlit python-dotenv

```


## Phase 2: Configuration & Data

### 5. Connect OpenAI

You must provide your own API key.

1. Get a key from [platform.openai.com](https://www.google.com/search?q=https://platform.openai.com/).
2. In your terminal, set it (replace `your_key_here` with your actual key):
```bash
export OPENAI_API_KEY="your_key_here"

```

#### 6. Ingest Your Course Material

Point the AI to your downloaded Canvas materials. This step "reads" your files and creates the
vector database (the `storage` folder). *Replace `[PATH_TO_YOUR_FILES]` with the actual folder path.*

```bash
python3 ingest.py [PATH_TO_YOUR_FILES]

```

> **Success Check:** You should now see a folder named `storage` in your directory.

---

## Phase 3: Launch

Run the following command to start the tutor:

```bash
streamlit run app.py

```

**Your browser will automatically open to `http://localhost:8501`.**

---





