# ToneMentor

## Project Overview

**ToneMentor** is an AI-powered API designed to help users, especially non-native English speakers, communicate more professionally in corporate environments. The system analyzes the tone of a given message, compares it to a professional standard, and determines if rewriting is necessary. If so, it can rephrase the message in a more professional tone.

---

## Phase 1: Tone Detection API

### Features

- **REST API with Flask:**  
  Provides endpoints for basic server health checks and tone prediction.

- **Tone Classification:**  
  Accepts a message via a POST request and returns a tone classification result using a machine learning model.

- **Modular Design:**  
  Code is organized for easy extension and future improvements.

### How It Works

1. **Send a POST request** to `/predict` with a JSON body containing a `text` field.
2. **The API analyzes** the tone of the message and returns the classification result in JSON format.

#### Example Request

```json
POST /predict
Content-Type: application/json

{
  "text": "Could you please send me the report by tomorrow?"
}
```

#### Example Response

```json
{
  "tone": "polite",
  "confidence": 0.92
}
```

---

## Phase 2: Professional Rewriting (Work in Progress)

- **Automated Rewriting:**  
  The next phase will enable the API to automatically rewrite messages that are not professional, enhancing clarity, tone, and formality and 
  sent along with the response .

- **User Feedback Loop:**  
  Users will be able to review and accept or reject suggested rewrites.

---

## Getting Started

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**
   ```bash
   python app.py
   ```

---

## Project Structure

```
tone-improver-api/
├── app.py
├── tone_classifier_utilites.py
├── response_generator.py
├── requirements.txt
└── README.md
```

---

## License

This project is for educational and research purposes.

---

## Roadmap

- [x] Phase 1: Tone detection API
- [ ] Phase 2: Professional rewriting (in progress)
