# **🔍 Automated Scientific Citation Verification**

## **📖 System Overview**

The system extracts claims and their sources and verifies their correctness using **Retrieval-Augmented Generation (RAG)**. It processes **scientific articles from arXiv.org** 📚, specifically those in **IEEE citation style**, ensuring consistent citation extraction. Only **claims with arXiv references** are supported.

## **📊 Ground Truth for Evaluation**

To accurately assess the system's performance, a well-defined **Ground Truth** is established based on strict classification rules:

### ✅ **Positive Cases**

- A claim is considered **correctly verified** if the system successfully retrieves its cited source from the document collection.

### ❌ **Negative Cases**

- Claims from a field **unrelated** to the main document (e.g., Physics claims in a Computer Science article) serve as **negative examples**.
- These claims should **not** be supported by the document collection, ensuring that the system does not incorrectly attribute sources.

This classification method forms the basis for a structured evaluation using key performance metrics, including **Precision, Recall, Accuracy, and F1-Score**.

## **📌 System Evaluation (Test Cases 16-21)**

The system was tested using **Test Cases 16–21**. The following performance metrics were calculated:

- 🎯 **Precision**: 91.97%  
- 🔄 **Recall**: 59.51%  
- 📈 **Accuracy**: 75.26%  
- ⚖️ **F1-Score**: 71.92%  

### **🏆 Conclusion**

The results demonstrate that **RAG-based citation verification is a viable approach**, effectively identifying and validating scientific claims. However, improvements in **Recall** are necessary to enhance the system’s ability to comprehensively verify citations.  🚀

## 🎯 Workflow

1. 📑 Extract claims and references from scientific papers
2. ⬇️ Download referenced papers from arXiv
3. 🔄 Preprocess claims to generate optimal search queries
4. 🔍 Retrieve relevant document sections using similarity search
5. ✅ Verify claims against the referenced papers
6. 📊 Evaluate verification results

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- LangChain API key (optional)

### Installation

1. **Clone the repository** 

   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   To properly configure the system, environment variables must be set in a .env file. 

## 💻 Usage

```bash
python main.py <testcase_name> [options]
```

### 🔧 Available Options

| Option | Description |
|--------|-------------|
| `--run [arxiv_id]` | Run a basic workflow (optional arXiv ID for new testcase) |
| `--create <arxiv_id>` | Create new testcase from arXiv paper |
| `--extract` | Extract claims from PDF |
| `--download-papers` | Download referenced papers |
| `--preprocess` | Generate search queries |
| `--process` | Run RAG verification |
| `--evaluate <results_dir>` | Generate metrics |

### 📋 Example Workflow

Use this example to create a new testcase from an arXiv paper: https://arxiv.org/pdf/2310.02368v1

```bash
python -m src.main my_testcase --run 2310.02368v1
```

or each step separately:

```bash
# Create new testcase
python -m src.main my_testcase --create 2310.02368v1

# Extract claims and refrences
python -m src.main my_testcase --extract

# Download references
python -m src.main my_testcase --download-papers

# Preprocess claims
python -m src.main my_testcase --preprocess

# Run verification
python -m src.main my_testcase --process

# Evaluate results
python -m src.main my_testcase --evaluate [results_dir]
```

## 📁 Output Structure

* **data/**
  * `logs/` - System logs
  * `negative_claims/` - Claims for negative cases
  * **testcase_name/**
    * `db/` - Vector store
    * `logs/` - Verification processing logs
    * `results/` - Verification results
    * `evaluation/` - Metrics & analysis
    * `referenced_papers/` - Downloaded PDFs
    * `extracted_data/` - Processed claims

The repository includes 21 example test cases (tc1 - tc21) and a minimal test case 'testcase_code_short' for rapid code validation.

## 🏗️ Project Structure

- **services/** - Core processing services
  - `testcase_service.py` - Test case creation and management
  - `extract_paper_data.py` - Claim and Reference extraction
  - `download_references.py` - arXiv paper downloader
  - `claim_preprocessor.py` - Query generation
  - `claim_processor.py` - RAG-based claim verification
  - `evaluator.py` - Results evaluation
  - `vector_store.py` - Vector store management
  - `langchain.py` - LangChain functionality

- **utils/** - Utility functions
  - `doc_utils.py` - Document handling
  - `path_utils.py` - Path management
  - `logger_mixin.py` - Logging mixin
  - `logger_md.py` - Markdown logging
  - `json_manager.py` - JSON file handling
  - `arxiv_utils.py` - arXiv functionality

- **prompts/** - LLM prompts
  - `rag_prompts.py` - prompts

- **loaders/** - Data loading
  - `pdf_loader.py` - PDF parsing

- **config/** - Configuration
  - `env_loader.py` - Environment variable loading
  - `system_config.yaml` - System configuration (Model, LangChain-Config, etc.)
  - `config_loader.py` - System configuration loading

- **models/** - Data models
  - `models.py` - Data models

- **scripts/** - Scripts
  - `extract_claims.py` - Claim extraction from multiple PDFs

- `main.py` - CLI entry point
- `requirements.txt` - Dependencies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
Made with ❤️ for scientific integrity
</div>