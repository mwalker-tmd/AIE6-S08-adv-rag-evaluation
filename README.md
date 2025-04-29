# AIE6-S08-Advanced-RAG-Evaluation

This project demonstrates how to evaluate and improve a Retrieval-Augmented Generation (RAG) application using LangGraph, LangChain, and RAGAS. It includes baseline and semantically-enhanced RAG pipelines, with detailed metric-based evaluation using RAGAS.

## ✅ Requirements For This Bonus Challenge:
##### **MINIMUM REQUIREMENTS**:

1. Baseline `LangGraph RAG` Application using `NAIVE RETRIEVAL`
2. Baseline Evaluation using `RAGAS METRICS`
  - [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
  - [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)
  - [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)
  - [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)
  - [Answer Correctness](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness.html)
3. Implement a `SEMANTIC CHUNKING STRATEGY`.
4. Create an `LangGraph RAG` Application using `SEMANTIC CHUNKING` with `NAIVE RETRIEVAL`.
5. Compare and contrast results.

##### **SEMANTIC CHUNKING REQUIREMENTS**:

Chunk semantically similar (based on designed threshold) sentences, and then paragraphs, greedily, up to a maximum chunk size. Minimum chunk size is a single sentence.


## 🔧 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/AIE6-S08-adv-rag-evaluation.git
   cd AIE6-S08-adv-rag-evaluation
   ```

2. **Create and activate environment**

   You can use either `uv` or `conda`.

   - With `uv`:
     ```bash
     uv venv
     source .venv/bin/activate
     uv pip install -r requirements.txt
     ```

   - Or with `conda`:
     ```bash
     conda env create -f environment.yml
     conda activate rag-env
     ```
    > ⚠️ **MacOS Users (pre-14.0)**: We recommend using `conda` to create the environment due to known issues with `faiss-cpu` and other packages failing to compile using `uv` or `pip`. Start with:
    >
    > ```bash
    > conda create -n rag-env python=3.10
    > conda activate rag-env
    > pip install -r requirements.txt  # or rely on environment.yml
    > ```

3. **Set environment variables**

   Copy `.env.example` to `.env` and update it with your OpenAI key:

   ```bash
   cp .env.example .env
   ```

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## ▶️ How to Run

Launch Jupyter Lab:

```bash
jupyter lab
```

Then open and run:

- `04_evaluate_baseline_vs_semantic.ipynb` → evaluates RAG outputs using RAGAS metrics.
- `05_compare_and_visualize_results.ipynb` → compares the metrics visually and prints a summary.

> 📂 Make sure the documents in `data/raw/sample_docs.txt` and evaluation questions in `data/raw/eval_questions.txt` are available and properly formatted.

---

## 📊 Evaluation Metrics

Evaluations use the following [RAGAS](https://github.com/explodinggradients/ragas) metrics:

- `faithfulness`
- `context_precision`
- `answer_relevancy`
- `answer_correctness`
- `context_recall` (may be NaN when ground truths are unavailable)

---

## 🧠 Developer Notes

- RAG apps are implemented in:
  - `src/rag/baseline_rag_app.py`
  - `src/rag/semantic_rag_app.py`

- Evaluation logic is in:
  - `src/evaluation/evaluate_baseline_rag.py`
  - `src/evaluation/evaluate_semantic_rag.py`

- Graph nodes and state schema are under `src/graph/`

- Custom semantic chunker logic is implemented in `src/retrieval/semantic_chunker.py`

---

## ✅ Status

- ✅ Baseline and semantic apps both run cleanly
- ✅ RAGAS evaluations execute with valid metrics
- ✅ Semantic chunking implemented (basic sentence-level)
- ⏳ Context precision could be further tuned

---

## 🙌 Acknowledgments

This project was built as part of [AI Makerspace's AI Engineering Bootcamp](https://aimakerspace.io/the-ai-engineering-bootcamp/) Cohort #6 — Session 08: Evaluating RAG With Ragas.
