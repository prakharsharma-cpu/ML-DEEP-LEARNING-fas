🦺 SmartPPE — Computer Vision–Based PPE Detection System

A Machine Learning & Deep Learning Project

📌 Project Overview

SmartPPE is an AI-powered computer vision system designed to automatically detect whether workers on a construction site are wearing the required Personal Protective Equipment (PPE) — such as hard hats, safety vests, masks, gloves, boots, goggles, and coveralls.

Using deep learning models and real-time image analysis, the system classifies each worker as:

✅ Compliant
🟡 Partially Compliant
❌ Non-Compliant

The goal is to improve safety, reduce manual inspection effort, and enable faster response to safety risks on construction sites.

🎯 Objectives

Automate PPE detection using computer vision

Reduce manual inspection errors

Provide real-time compliance monitoring

Improve workplace safety & regulatory compliance

Deploy an easy-to-use Streamlit interface

🚀 Project Workflow
Step 4 — Model Selection

Choosing the appropriate ML/DL model (YOLO-based architecture).

Step 5 — Model Design & Training

Building the AI system:

Data preprocessing

Annotation

Training on PPE dataset

Hyperparameter tuning

Step 6 — Model Evaluation & Testing

Evaluating accuracy, precision, recall & prediction robustness.

Step 7 — Deployment with Streamlit

Creating a visual dashboard that:

Shows detections

Displays compliance color codes

Generates reports

Step 8 — Monitoring & Maintenance

Continuous improvement using real-world feedback.

📂 Dataset Description

The dataset contains thousands of real construction-site images labeled in YOLO format.
Classes include:

Hardhat

No-Hardhat

Mask

No-Mask

Safety Vest

No-Safety Vest

And other PPE classes

Images cover multiple lighting conditions, angles, worker postures, and environments to improve robustness.

🖼 Sample Outputs

(Add prediction images here)

📊 Final Evaluation Metrics

Example model output:

{'person_id': 1, 'present': [], 'missing': ['helmet', 'vest', 'mask', 'goggles', 'gloves', 'boots', 'coverall'], 'status': 'RED'}

{'person_id': 2, 'present': [], 'missing': ['helmet', 'vest', 'mask', 'goggles', 'gloves', 'boots', 'coverall'], 'status': 'RED'}

🎬 Recommended Video Script (Opening & Closing Scenes)
✅ Opening Scene Ideas

Real construction site shots

Workers with proper and improper PPE

Supervisors checking PPE manually

Introduction to SmartPPE and its purpose

✅ Closing Scene Ideas

Demonstration of model detecting PPE

Streamlit dashboard showing compliance

Impact on safety & accident reduction

Final credits / acknowledgments

🔗 Useful Links

You can replace these with your actual links:

GitHub Repository:
https://github.com/yourusername/yourrepo

Google Colab Notebook:
https://colab.research.google.com/...

🛠 Tech Stack

Python

YOLO (Ultralytics)

OpenCV

NumPy

Streamlit

Deep Learning (CNNs)

🤝 Contributions

Pull requests are welcome!

📜 License

This project is licensed under the MIT License.
