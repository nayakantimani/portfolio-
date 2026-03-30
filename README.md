# 🤖 Therapy Connect – RAG + Emotion Detection + Audio Processing

## 📌 Overview

This project is an advanced AI pipeline that integrates **emotion detection, audio processing, and Retrieval-Augmented Generation (RAG)** to analyze user input and generate intelligent responses.

It is designed as a **modular, scalable backend system** using FastAPI, enabling real-time interaction with AI models.

---

## 🚀 Features

* 🎤 **Audio Processing** using FFmpeg
* 😊 **Emotion Classification** from user input
* 🧠 **RAG-based LLM System** for intelligent responses
* ⚡ **FastAPI Backend** for real-time API interaction
* 📦 Modular pipeline architecture for scalability
* 🔄 Session-based processing and summarization

---

## 🛠️ Tech Stack

* **Languages:** Python
* **Backend:** FastAPI
* **AI/ML:** NLP, Emotion Detection Models, RAG (LLM Integration)
* **Data Processing:** Kafka (concept), Streaming-ready pipeline
* **Audio Processing:** FFmpeg
* **Database:** Vector DB (for knowledge retrieval)

---

## 🏗️ Project Structure

```bash
components/        # Core reusable modules  
models/            # ML / AI models  
pipeline_io/       # Input/Output handling  
services/          # API services and orchestration  
templates/         # Prompt templates / configs  
transcriptions/    # Audio/text outputs  
data/              # (Excluded large files)  
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/my-ai-pipeline
cd my-ai-pipeline

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
uvicorn main_app:app --reload --port 8000
```

---

## 🌐 API Usage

Open Swagger UI:

👉 http://127.0.0.1:8000/docs

### Available Endpoints:

* **POST /start_session** → Initialize AI session
* **POST /get_summary** → Generate AI-based summary

---

## 📊 Example Workflow

1. Start session using `/start_session`
2. Provide input (text/audio)
3. System processes:

   * Emotion detection
   * Knowledge retrieval (RAG)
   * LLM response generation
4. Get summarized output via `/get_summary`

---

## 📦 Data & Model Files

Large data files (vector databases, embeddings, and model artifacts) are not included due to GitHub size limitations.

To run the project:

* Create or integrate your own dataset
* Place required files in:

```bash
data/knowledge_base_db/
```

---

## 🔥 Key Highlights

* End-to-end AI pipeline implementation
* Combines NLP, audio processing, and LLMs
* Designed for real-world scalable applications
* Demonstrates backend + AI integration

---

## 🚀 Future Enhancements

* Add frontend UI (React / Streamlit)
* Deploy on AWS / Docker
* Integrate advanced LLMs (GPT / Transformers)
* Real-time streaming with Kafka

---

## 👨‍💻 Author

**Mani Shankar Reddy**
AI/ML Engineer | Data Engineer | Backend Developer

---

⭐ *If you like this project, feel free to star the repository!*
