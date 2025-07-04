﻿## Rahii.ai

**Rahii.ai** is a full-stack AI assistant application built using FastAPI for the backend and React for the frontend. It supports text-to-speech (TTS), speech-to-text (ASR), and multilingual capabilities with smart handling of Indic languages like Odia, Hindi, and more.

---

### Tech Stack

| Layer            | Technology                     |
| ---------------- | ------------------------------ |
| Frontend         | React (Vite or CRA)            |
| Backend          | FastAPI (Python 3.9+)          |
| TTS Engine       | SarvamAI, Edge TTS             |
| ASR Engine       | Whisper by OpenAI              |
| Package Managers | npm, pip                       |
| Language Support | English, Hindi, Odia, and more |

---

### Features

* Text-to-Speech using SarvamAI and Edge TTS
* Automatic selection of TTS engine based on language
* Whisper-based transcription support
* React-based frontend for user interaction
* FastAPI backend with async APIs
* One-click development startup script
* Scalable and modular architecture

---

### Project Structure

```
Rahii.ai/
├── Backend/                 # FastAPI app
│   ├── main.py
│   ├── requirements.txt
│   └── ...
├── Frontend/               # React app
│   ├── package.json
│   ├── src/
│   └── ...
├── run.py                  # Dev runner script
├── .gitignore
└── README.md
```

---

### Setup Instructions

#### Backend (FastAPI)

```bash
cd Backend
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend (React)

```bash
cd Frontend
npm install
npm run dev
```

---

### Quick Start

Use the `run.py` script from the project root to set up and run both the frontend and backend:

```bash
python run.py
```

This will:

* Install dependencies in `Frontend/` and `Backend/`
* Launch the React app at `http://localhost:3000`
* Launch the FastAPI app at `http://127.0.0.1:8000/docs`

---

### Language Support

| Language | Code | TTS Engine        |
| -------- | ---- | ----------------- |
| English  | en   | Edge TTS          |
| Hindi    | hi   | Edge TTS          |
| Odia     | or   | SarvamAI (Swayam) |
| Marathi  | mr   | Edge TTS          |
| Tamil    | ta   | Edge TTS          |
| Bengali  | bn   | Edge TTS          |

---

### Environment Variables


Set your SarvamAI API key and Groq API key using an environment variable:

```bash
export SARVAM_API_KEY=your-api-key
export GROQ_API_KEY=your-api-key
```

Or place it in a `.env` file if using `python-dotenv`.

---

### Contributing

Contributions are welcome. Please open an issue to discuss what you would like to change or improve.

---

### License

This project is licensed under the MIT License.

---

