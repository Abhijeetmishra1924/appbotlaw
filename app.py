# app.py
# ✅ Vidur Bot: Flask + LangChain + Groq Backend (Multi-language + PDF Upload)

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, urllib.parse, requests
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf

# Load API keys from environment (Render supports this)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Directory setup
data_dir = "data"
persist_path = "chroma_db"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(persist_path, exist_ok=True)

# Initialize LLM
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# Setup Vector DB (RAG)
if not os.path.exists(os.path.join(persist_path, "chroma.sqlite3")):
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_path)
    vector_db.persist()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=persist_path, embedding_function=embeddings)

retriever = vector_db.as_retriever()

# YouTube fetcher
def fetch_youtube(query):
    query += " According to law India"
    url = (
        f"https://www.googleapis.com/youtube/v3/search?"
        f"part=snippet&maxResults=5&q={urllib.parse.quote(query)}"
        f"&key={YOUTUBE_API_KEY}&type=video&regionCode=IN"
    )
    try:
        response = requests.get(url)
        items = response.json().get("items", [])
        return [
            f"[{item['snippet']['title']}](https://www.youtube.com/watch?v={item['id']['videoId']})"
            for item in items
        ]
    except:
        return []

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "✅ Vidur Bot is running. Use the /ask endpoint via POST to interact."

# PDF upload
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(data_dir, filename)
    file.save(file_path)
    try:
        reader = pypdf.PdfReader(file_path)
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return jsonify({"pdf_text": extracted_text})
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

# Ask endpoint
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_input = data.get("question", "")
        pdf_text = data.get("pdf_text", "")
        user_lang = data.get("lang", "English")

        if pdf_text:
            user_input = (
                "The user uploaded the following legal document:\n\n"
                + pdf_text
                + "\n\nPlease explain the content and what the user should do."
            )

        prompt_template = PromptTemplate(
            template=f"""Act as a law expert in India named Vidur Bot. 
Answer queries with deep understanding of Indian law, quoting Constitution Articles and BNS where relevant. 
Give remedies and user rights. Reply in {user_lang}.

{{context}}

User: {{question}}
Vidur Bot:""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )

        response = qa_chain.run(user_input) or "Sorry, I couldn't generate a response."
        youtube = fetch_youtube(user_input)
        return jsonify({
            "message": response,
            "youtube_links": youtube
        })
    except Exception as e:
        return jsonify({
            "message": f"An error occurred: {str(e)}",
            "youtube_links": []
        })

# ✅ Final Render-compatible server start
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
