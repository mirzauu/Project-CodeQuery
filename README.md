# Repoplay

<p align="center">
  <img src="app/assets/LOGO.png" alt="Repoplay Logo" width="120" height="120" style="border-radius: 50%; object-fit: cover;" />
</p>

**Repoplay** is an AI-Powered Git Repository Assistant that allows users to connect any public GitHub repository and interact with it using natural language. Simply input the repo URL, and Repoplay helps you explore, understand, and query the project using powerful language models and graph-based reasoning.

🎥 **Demo Video**  
[![Watch the demo](link-to-demo-thumbnail.jpg)](https://link-to-your-demo-video.com)

---

## 🚀 Features

- 🔗 Connect any **public GitHub repository**
- 🤖 Ask **natural language questions** about repo code, structure, and logic
- 🧠 Uses **LLMs + LangChain** for contextual reasoning
- 🕸️ Employs **Neo4j Graph Database** for structural representation
- ⚡ Built with **FastAPI** for speed and performance
- 📊 Combines **PostgreSQL + MongoDB** for scalable data storage
- 🔐 JWT-based authentication with email support

---

## 🧩 Tech Stack

- **Backend**: FastAPI, Python 3.10  
- **AI/LLM Integration**: PydanticAI, LangChain  
- **Databases**: Neo4j, PostgreSQL, MongoDB  
- **Auth & Email**: JWT, SMTP

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/repoplay.git
```

### 2. Create and configure `.env`

```ini
# MongoDB Configuration
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority&appName=<app-name>
MONGO_DB_NAME=your_db_name

# PostgreSQL Configuration
SQLALCHEMY_DATABASE_URL=postgresql+psycopg2://<username>:<password>@<host>:<port>/<database>

# SMTP Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_email_app_password
EMAIL_FROM=your_email@gmail.com

# Neo4j Configuration
NEO4J_URI=neo4j+s://<your-neo4j-host>
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

# LLM API Key
LLM_API_KEY=your_llm_api_key

# Authentication
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=300
```

### 3. Create a Python 3.10 environment and install dependencies

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn app.main:app --reload
```

---

## 📚 API Documentation

Interactive API docs available at:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🔐 Authentication Endpoints

#### ✅ Send OTP  
`POST /api/v1/auth/send-otp`

**Request Body:**
```json
{
  "email": "user@example.com",
  "first_name": "string",
  "last_name": "string"
}
```

**Response (200):**
```json
"string"
```

#### ✅ Verify OTP  
`POST /api/v1/auth/verify-otp`

**Request Body:**
```json
{
  "email": "user@example.com",
  "otp": "string"
}
```

**Response (200):**
```json
"string"
```

---

### 📦 Repository Parsing

#### 🔍 Parse Repository  
`POST /api/v1/auth/parse`

**Request Body:**
```json
{
  "repo_link": "string"
}
```

**Response (200):**
```json
"string"
```

#### 📂 Get Projects  
`GET /api/v1/auth/projects`

**Response (200):**
```json
"string"
```

---

### 💬 Conversations

#### 💡 Ask a Question  
`POST /api/v1/auth/conversations/{project_id}/message/`

**Path Parameter:**
- `project_id` (string)

**Request Body:**
```json
{
  "content": "string"
}
```

**Response (200):**
```json
"string"
```

#### 📜 Get Conversation History  
`GET /api/v1/auth/conversations/{project_id}/history`

**Path Parameter:**
- `project_id` (string)

**Response (200):**
```json
"string"
```


Interactive documentation available at:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

### Example Endpoints

#### 🔐 Register User

- `POST /auth/register`

```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

#### 🔐 Login

- `POST /auth/login`

```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

#### 🔗 Connect GitHub Repo

- `POST /repo/connect`

```json
{
  "repo_url": "https://github.com/owner/repo"
}
```

#### ❓ Ask a Question

- `POST /repo/query`

```json
{
  "repo_id": "abc123",
  "question": "What is the purpose of the utils.py file?"
}
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Make sure to include tests as appropriate.

---
