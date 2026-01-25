# 🔍 CLIP-based Semantic Search Engine

An AI-powered image search application that allows users to search for images using natural language queries. Built with **OpenAI's CLIP model**, **PyTorch**, and **Streamlit**.

Unlike traditional keyword search, this system understands the *content* and *context* of images, enabling queries like "a dog running in the grass" or "delicious food" without requiring any image tags or labels.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CLIP-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## ✨ Features

* **Semantic Search:** Search for images using descriptive text (Text-to-Image retrieval).
* **Zero-Shot Learning:** No model training required; uses pre-trained CLIP embeddings.
* **Auto-Indexing:** Automatically downloads a test dataset (via Lorem Picsum) and builds a vector index if one doesn't exist.
* **Robust Downloader:** Includes a custom image downloader that handles network errors and bypasses bot detection (403/429 errors).
* **Interactive UI:** Clean and responsive web interface using Streamlit.

## 📂 Project Structure

```bash
project-4/
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Python dependencies
├── src/
│   ├── config.py       # Configuration settings (paths, device)
│   ├── model.py        # CLIP model loader and embedding logic
│   └── indexer.py      # Image downloader and vector indexing engine
└── data/
    ├── images/         # Directory for downloaded images
    └── embeddings.pt   # Saved vector embeddings (cache)
```

## 🚀 Installation & Setup
This project is optimized for Windows environments to avoid common DLL/GPU errors.

1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd project-4
```
2. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to keep dependencies clean.

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

3. Install Dependencies
Note for Windows Users: To prevent OSError: [WinError 1114], install the CPU version of PyTorch first.

```Bash
# 1. Install PyTorch (CPU Version)
python -m pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```
```bash
# 2. Install other requirements
python -m pip install transformers pillow streamlit numpy requests tqdm
```
🏃‍♂️ How to Run
Execute the application using the following command:

```bash
python -m streamlit run app.py
```
What happens next?

The app will check for images in data/images.

If empty, it will automatically download 20 random high-quality images from Lorem Picsum.

It will generate embeddings (vector representations) for these images using CLIP.

The Web UI will open in your browser (usually http://localhost:8501).

## 🖼️ Usage Example
Wait for the indexing process to finish (indicated by the progress bar).

In the search bar, type a query like:

"A peaceful mountain"

"Red color object"

"Cute animal"

The system will display the most relevant images ranked by similarity score.

## 🔧 Troubleshooting
```bash
OSError: [WinError 1114] DLL initialization failed
This occurs when the GPU version of PyTorch conflicts with Windows system DLLs.
```
Fix: Uninstall the current torch version and reinstall the CPU-specific version using the command provided in the Installation section.

```bash
403 Forbidden or 429 Too Many Requests when downloading images
```
Fix: The project uses src/indexer.py with a custom User-Agent header and utilizes the Lorem Picsum API to ensure stable image downloading without being blocked by servers.

## 📚 Tech Stack
Language: Python

Model: OpenAI CLIP (via Hugging Face Transformers)

Framework: PyTorch

UI: Streamlit

Data Handling: PIL (Pillow), NumPy

## 📝 License
This project is open-source and available under the MIT License.