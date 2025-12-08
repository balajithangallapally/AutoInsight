# 🚀 AutoInsight – Personal Data Analytics Dashboard

AutoInsight is a **full-stack web application** that transforms CSV activity logs into insightful analytics, summaries, visualizations, and trend forecasts.  
It is built with a modern stack: **FastAPI** (backend) + **React.js** (frontend) + **SQLite** (storage).

This project is designed for hands-on learning in **web development**, **data engineering**, **API design**, and **data visualization**, making it an excellent addition to a resume or hackathon submission (Virtusa Neural Hackathon 2025 ready).

---

## ✨ Features

### 🔹 Data Upload & Processing
- Upload any CSV dataset through the UI  
- Automatic parsing using Pandas  
- Detects dates, categories, numbers, missing values  
- Stores analysis history in SQLite  

### 🔹 Visual Analytics Dashboard
- Time-series plot (activity over time)  
- Category distribution pie-chart  
- Numeric column statistics  
- Clean, responsive UI built with React + Chart.js  

### 🔹 Backend API (FastAPI)
- `/api/upload` – upload and analyze CSV  
- `/api/reports` – list previous reports  
- `/api/reports/{id}` – fetch a specific analysis  
- Auto-generated API docs at `/docs`  

### 🔹 Database
- SQLite database to store reports  
- Persistent between sessions  

### 🔹 Extensible & ML-Ready
You can integrate:
- Anomaly detection  
- Forecasting models  
- Recommendation systems  
- NLP summaries  
(ML code scaffolding included in backend.)

---

## 🛠️ Tech Stack

### **Frontend**
- React.js (Create React App)
- Axios (HTTP client)
- Chart.js & react-chartjs-2 (visualizations)
- Tailwind or custom CSS (optional)

### **Backend**
- FastAPI (Python)
- Pandas
- SQLite
- Uvicorn server
- Python-Multipart (file uploads)

### **Other Tools**
- Git / GitHub
- Virtualenv
- Node.js & npm
- OpenAI API (optional future integration)

---

## 📂 Project Structure

