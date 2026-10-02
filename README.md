# Medical AI Intelligence Platform

A high-performance clinical reasoning platform providing medical diagnosis, drug pharmacology intelligence, multimodal vision analysis, and real-time AI consultations.

Developed by **Sadhana Patel**.

---

## ✨ Features & Capabilities

- **SQLite Email Authentication**: Secure email and password login system with Werkzeug cryptographic hashing and isolated user sessions.
- **Pydantic Validation**: Robust, type-safe request parsing and friendly field validation on all medical workflows.
- **Gemini 2.5 Flash Intelligence**: Powered by Google's flagship multimodal model (`models/gemini-2.5-flash`), delivering ultra-fast clinical differential diagnoses, pharmacological breakdowns, and vision scan analysis.
- **Multimodal Medical Vision Scanner**: Radiologic interpretation for chest and skeletal X-rays, melanoma/dermatology lesion classification, tumor screening, and custom clinical queries.
- **Drug Intelligence Suite**: Search medications by generic substance or commercial brand name across international pharmacopeia standards (US, UK, EU, India, Canada, Australia).
- **Interactive Molecular Consultation**: AI chat supporting biochemical inquiries, target ligands, and binding property simulations.
- **Encrypted History Vault**: Patient records and AI reports are securely isolated per user and downloadable on-demand.
- **Sleek, Modern UI**: Designed with glassmorphism, responsive navigation, micro-interactions, custom generated medical visuals, and live telemetry badges.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Flask (Python 3)
- **Validation**: Pydantic v2 & `email-validator`
- **Database**: SQLite 3 with user schema migration
- **AI / LLM**: Google Gemini 2.5 Flash (`google-generativeai`)
- **Vision / Imaging**: Pillow (PIL)
- **Frontend**: Custom Modern Design System (Vanilla CSS, Plus Jakarta Sans, Inter, FontAwesome 6)
- **Security**: Cryptographic password hashing (`scrypt` / `pbkdf2`), server-side secure cookies

---

## 🚀 Getting Started

### 1. Clone & Set Up Directory
```bash
git clone https://github.com/yourusername/medical-ai-assistant.git
cd medical-ai-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create or verify the `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=models/gemini-2.5-flash
SECRET_KEY=medical_ai_platform_sadhana_patel_sec_2026
```

### 4. Run the Platform
```bash
python app.py
```
Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

---

## 👩‍💻 Lead Developer & Engineering
- **Developer**: [Sadhana Patel](https://www.linkedin.com/in/sadhana-patel-9baab82b0)
- **LinkedIn**: [Sadhana Patel](https://www.linkedin.com/in/sadhana-patel-9baab82b0)
- **Platform**: Metropolitan General Hospital Clinical AI Decision Workstation
- **Year**: 2026

---

## 🔒 Clinical Safety Notice
This application is designed for educational, research, and clinical decision-support purposes. It is not intended to replace professional judgment or direct physician diagnosis.