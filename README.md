# 🛡️ Email Threat Classifier (Gmail Inbox Agent)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=flat&logo=mongodb&logoColor=white)
![Jenkins](https://img.shields.io/badge/Jenkins-CI/CD-red?logo=jenkins)
![AI](https://img.shields.io/badge/AI-RoBERTa%20%7C%20Gemma%202B-orange)

An automated AI pipeline that acts as a cybersecurity analyst for your inbox. It continuously polls unread emails, classifies them for potential threats using a dynamic multi-stage machine learning pipeline, and automatically quarantines malicious content using native Gmail labels.

## 📖 Overview

This agent actively secures your Gmail inbox by filtering out threats before you even see them. It categorizes emails into explicit labels: `Clean`, `Spam`, `Phishing`, `Fraud`, `Malware`, or `Threat`. Every action, along with the AI's detailed inference reasoning, is persistently logged to a local MongoDB database for auditing and continuous learning.

## 🏗️ Architecture: Model Context Protocol (MCP)

To ensure high performance and prevent I/O bottlenecks from blocking GPU resources, the system architecture is decoupled into two primary services using the Model Context Protocol:

1. **MCP Model Server (`model_server.py`)**: A standalone, high-performance FastAPI service that keeps the heavy machine learning models warm in GPU memory. It exposes modular endpoints (`/classify`, `/classify/batch`) for rapid querying.
2. **Gmail Email Agent (`main.py`)**: A lightweight background polling service. It manages Gmail API interactions, payload extraction, API routing, and database logging without getting bogged down by model inference times.

## 🧠 Dynamic Inference Engine

The core inference engine (`orchestrator.py`) utilizes a smart, 3-component pipeline designed to perfectly balance inference speed with deep analytical accuracy:

1. **Fast Path (RoBERTa Classifier)**: A highly optimized, 6-class sequence classification model running on the GPU for rapid initial analysis.
2. **Open-Set Gate**: Applies temperature scaling and a confidence threshold (`tau`) to the RoBERTa output. High-confidence predictions bypass further processing, saving significant compute resources.
3. **Deep Path (Gemma 2B LoRA)**: If the initial confidence is below the threshold, the email is routed to a quantized (4-bit NF4) Gemma 2B LoRA model. Acting as a specialized cybersecurity analyst, it performs deep conversational analysis and returns a structured JSON payload detailing the computed label, confidence score, specific reasoning, extracted threat indicators, and recommended actions.

## ⚙️ Gmail API Integration

The agent acts autonomously to secure the user environment:
* **OAuth 2.0 Auth**: Secure authentication via `token.json`.
* **Dynamic Label Provisioning**: Automatically detects or creates necessary semantic folders within the user's Gmail UI.
* **Active Mitigation**: Continuously fetches unread messages, extracts nested MIME body content, and strips HTML. Malicious emails are automatically stripped of `UNREAD` and `INBOX` labels (quarantining them from the primary view) and assigned their specific threat label.

## 🚀 CI/CD Automation (Jenkins)

Continuous Integration and Deployment are fully automated via a robust Jenkins pipeline (`Jenkinsfile` & `script.groovy`):

* **Init & Build**: Provisions isolated Python virtual environments (`.venv`) and intelligently caches/installs dependencies.
* **Test & Quality Gate**: Executes a comprehensive `pytest` suite generating JUnit XML results and HTML coverage reports. The pipeline enforces a strict quality gate, halting deployment if test coverage drops below `60%`.
* **Linting**: Enforces PEP-8 standards and clean code practices via `flake8`.
* **Zero-Downtime Deploy**: Performs automated service handoffs by gracefully terminating legacy `uvicorn` processes and spinning up updated instances using `nohup` for seamless background execution.

## 🛠️ Setup & Installation

*(Add your local installation steps here, e.g., cloning the repo, setting up the `.env` variables, generating the `token.json` via Google Cloud Console, and running `docker-compose up` or the Jenkins pipeline).*
