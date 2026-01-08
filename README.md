# MoSPI Chatbot - Project Setup

## Quick Start

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
Download and install Ollama from: https://ollama.ai/

### 5. Install Llama3.2:3b Model
```bash
ollama pull llama3.2:3b
```

### 6. Run Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8095
```

The application will be available at: http://localhost:8095