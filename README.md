#  RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about their content using a conversational chat interface.

The application retrieves relevant passages from documents and uses a locally running language model to generate answers grounded in the retrieved context.

##  Features

- **PDF processing:** Extract text from PDF documents.
- **Text chunking:** Split extracted text into smaller overlapping chunks.
- **Semantic search:** Generate embeddings using Sentence Transformers.
- **Vector search:** Use FAISS to retrieve relevant document passages.
- **AI-powered answers:** Generate answers using Ollama and Llama 3.2.
- **Source citations:** Display source filenames, page numbers, and similarity scores.
- **Conversational interface:** Ask questions and follow up using previous conversation context.
- **Local execution:** Run the language model locally with Ollama.

##  Tech Stack

- Python
- Streamlit
- Ollama
- Llama 3.2 (`llama3.2:1b`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- PyMuPDF
- NumPy

##  Project Structure

```text
rag-document-assistant/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── storage/
```

The `data/` directory holds PDF documents, while `storage/` contains the generated FAISS index and document metadata. These generated and potentially private files are excluded from version control by default.

##  Setup and Installation

### 1. Install prerequisites

Install Python, Git, and Ollama on your computer.

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads
- Ollama: https://ollama.com/download

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-assistant.git
cd rag-document-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download the language model

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running before asking questions.

##  Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

### Using the application

1. Upload a PDF using the sidebar.
2. Click **Process PDF** to extract, embed, and index document text.
3. Ask a question about the uploaded content.
4. Review the answer and its source citations.
5. Ask follow-up questions in the same conversation.

##  Build the Index from the Command Line

To index PDF files placed in the `data/` directory:

```bash
python rag.py build
```

To ask a question from the terminal:

```bash
python rag.py ask --question "What is this document about?"
```

To inspect retrieved passages:

```bash
python rag.py search --question "What is this document about?"
```

##  Limitations

- Answer quality depends on PDF text extraction, retrieval relevance, and language-model performance.
- Scanned PDFs may require OCR if they do not contain selectable text.
- Follow-up question rewriting can occasionally produce an incorrect search query.
- Source citations indicate retrieved passages; they do not guarantee that every generated statement is supported.
- The current implementation searches across all indexed PDFs rather than strictly limiting retrieval to one selected document.
- The application is designed for local use and requires Ollama and the configured language model.

## Future Improvements

- Filter retrieval by the selected document.
- Improve retrieval quality and relevance evaluation.
- Add OCR support for scanned PDFs.
- Add automated tests and error handling.
- Improve answer verification and source attribution.
- Explore deployment options for hosted use.

##  License

No license has been specified yet. Add a license if you intend to permit others to reuse or modify this project.

---

**Built as a learning project exploring Retrieval-Augmented Generation, semantic search, vector databases, and local language models.**