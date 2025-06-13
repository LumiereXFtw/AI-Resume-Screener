# AI-Based Resume Screener

An intelligent and efficient resume screening and matching tool powered by Google Gemini API, Python Flask, and Node.js. This system automates the process of analyzing resumes against job descriptions, providing quick, data-driven insights and streamlining the talent acquisition workflow.

## 🌟 Features

* **Overall Match Score:** Calculates a clear percentage match between a candidate's resume and a specific job description, providing an immediate understanding of fit.
* **Detailed AI Analysis:** Leverages the Google Gemini API to generate comprehensive feedback, including:
    * Summary of candidate suitability.
    * Key strengths and areas for improvement.
    * Identified missing requirements.
    * Estimated experience level.
    * Recommended next steps for the recruitment process.
    * Estimated salary expectations.
    * Suggested interview questions.
* **Matched Skills Extraction:** Automatically identifies and lists skills present in the resume that directly align with the job requirements.
* **Comprehensive Candidate Information Extraction:** Efficiently pulls out key candidate details such as:
    * Years of experience.
    * Contact information (emails, phone numbers).
    * Online profiles (LinkedIn, GitHub).
    * Educational background.
* **User-Friendly Dashboard:** A clean and intuitive web interface for easy resume uploading, job description input, and visualization of all analysis results.
* **Robust File Handling:** Securely manages resume file uploads and storage via the Node.js backend.

## 🚀 Real-World Impact

* **Time-Saving:** Significantly reduces the manual effort and time spent sifting through numerous resumes.
* **Bias Reduction:** Helps minimize unconscious bias in the initial candidate review process by providing objective, AI-driven insights.
* **Improved Candidate Quality:** Enables recruiters to quickly identify and focus on the most qualified candidates.
* **Accelerated Hiring:** Speeds up the entire talent acquisition cycle, allowing faster engagement with top talent.
* **Scalability:** Designed to handle various job roles and a large volume of resumes, making it suitable for growing organizations.

## ⚙️ Tech Stack

This project is built using a microservices architecture with distinct components:

### **1. Python Backend (Flask API)**
* **Role:** The core intelligence layer. Handles resume parsing, NLP operations, similarity calculations, and integration with the Google Gemini API for deep AI analysis.
* **Key Technologies:**
    * `Flask`, `Flask-CORS`: For building the RESTful API and handling cross-origin requests.
    * `google-generativeai`: Python SDK for interacting with the Google Gemini API.
    * `spaCy`: Advanced Natural Language Processing library for entity extraction and semantic similarity. (Requires `en_core_web_sm` model).
    * `scikit-learn`: For TF-IDF vectorization and cosine similarity to quantify text relevance.
    * `pdfplumber`, `python-docx`: Libraries for extracting text content from PDF and Word document formats respectively.
    * `python-dotenv`: For loading environment variables (like API keys) securely.

### **2. Node.js Backend (Express.js)**
* **Role:** Acts as an intermediary for file uploads and management. It receives resumes from the frontend, securely stores them, and forwards them to the Python backend for processing. It also manages screening data storage (in-memory for this demo, but extendable to a database).
* **Key Technologies:**
    * `Express.js`, `CORS`: For building the web server and handling API routes.
    * `Multer`: Middleware for handling `multipart/form-data`, primarily for file uploads.
    * `Axios`: Promise-based HTTP client for making requests to the Python API.
    * `dotenv`: For loading environment variables.
    * `form-data`, `fs`, `path`, `mime-types`, `uuid`: For file system operations, generating unique IDs, and handling file types.

### **3. Frontend (HTML, CSS, JavaScript)**
* **Role:** The user interface for interacting with the resume screener. It handles resume and job description input, displays real-time loading indicators, and presents the analysis results in a clear, card-based dashboard.
* **Key Technologies:**
    * HTML5: Structure of the web page.
    * CSS3: Styling for a modern, responsive, and user-friendly design.
    * JavaScript: Handles all dynamic interactions, form submission, API calls, and rendering of the analysis results.

## 🛠️ Setup and Running Locally

Follow these steps to get the AI-Based Resume Screener up and running on your local machine.

### **1. Prerequisites**

Before you begin, ensure you have the following installed:

* **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
* **Node.js LTS (recommended)**: [Download Node.js](https://nodejs.org/en/download/) (includes npm)
* **Google Gemini API Key**: [Get an API Key](https://ai.google.dev/gemini-api/docs/api-key)

### **2. Clone the Repository**

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/LumiereXFtw/AI-Resume-Screener.git

3. Environment Configuration (.env files)
You'll need to set up environment variables for your API keys and service URLs.

Create a file named .env in the root directory of your project. This file will store your actual secrets and is ignored by Git.

Code snippet

# .env
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY_HERE"
PYTHON_API="http://localhost:5001/api/screen"
FLASK_PORT=5001
FLASK_DEBUG=True
Replace "YOUR_GOOGLE_GEMINI_API_KEY_HERE" with the API key you obtained from Google AI Studio.

# .env.example
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
PYTHON_API="http://localhost:5001/api/screen"
FLASK_PORT=5001
FLASK_DEBUG=True

4. Set up Python Backend
Navigate into your project root directory if you're not already there.

Create a Virtual Environment: It's highly recommended to use a virtual environment to manage dependencies.
Bash

python -m venv venv
Activate the Virtual Environment:
On Windows:
Bash

.\venv\Scripts\activate
On macOS / Linux:
Bash

source venv/bin/activate
Install Python Dependencies: Create a requirements.txt file in your project root with the following content:
# requirements.txt
Flask==2.3.2
Flask-Cors==4.0.0
pdfplumber==0.10.1
python-docx==1.1.0
spacy==3.7.4
scikit-learn==1.3.2
google-generativeai==0.3.0
python-dotenv==1.0.1
Then, install them:
Bash

pip install -r requirements.txt
Download spaCy Model: The en_core_web_sm model is required for spaCy.
Bash

python -m spacy download en_core_web_sm
5. Set up Node.js Backend (Optional Proxy/File Handler)
Navigate into your project root directory.

Create package.json: If you don't already have a package.json file, create one with the following content:
JSON

// package.json
{
  "name": "resume-screener-node",
  "version": "1.0.0",
  "description": "Node.js backend for AI-based Resume Screener (handles file uploads, storage, and analytics data).",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "@google/generative-ai": "^0.11.0",
    "axios": "^1.6.8",
    "cors": "^2.8.5",
    "dotenv": "^16.4.5",
    "express": "^4.18.2",
    "form-data": "^4.0.0",
    "mime-types": "^2.1.35",
    "multer": "^1.4.5-lts.1",
    "uuid": "^9.0.1"
  }
}
Install Node.js Dependencies:
Bash

npm install
# OR
yarn install
6. Run the Applications
You'll need to run both the Python and Node.js backends in separate terminal windows.

Start Python Backend:
Open a new terminal, activate your Python virtual environment, and run:

Bash

# Ensure you are in the project root and venv is active
python app.py
This will start the Python Flask API, typically on http://localhost:5001.

Start Node.js Backend (Optional):
Open another new terminal (do NOT activate the Python venv here, just ensure you are in the project root) and run:

Bash

npm start
# OR
node server.js
This will start the Node.js Express server, typically on http://localhost:5000.

Open the Frontend Dashboard:
Once both backends are running, open the index.html file located in your project's root directory directly in your web browser. The frontend is configured to call the Python backend at http://localhost:5001.

Alternatively, if you have your Node.js server configured to serve static files, you might access it via http://localhost:5000 (or whatever port your Node.js server is listening on).

🚀 Usage
Select Resume: Click the "Choose File" button under "Upload Resume & Job Description" to select a PDF, DOCX, or TXT resume file.
Enter Job Description: Paste the relevant job description into the provided text area.
Analyze: Click the "Analyze Resume" button.
View Results: The dashboard will update with:
An Overall Match Percentage.
A comprehensive Gemini AI Analysis of the candidate's fit.
A list of Matched Skills found in the resume relevant to the job.
Candidate Information extracted from the resume.
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature or fix/BugFix).
Make your changes.
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request.
📜 License
This project is open-source and available under the MIT License.
