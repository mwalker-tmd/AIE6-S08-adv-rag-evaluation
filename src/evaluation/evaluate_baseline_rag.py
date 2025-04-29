from configs.evaluation_config import RAGAS_METRICS
from src.rag.baseline_rag_app import baseline_rag_app
from ragas import evaluate
from datasets import Dataset
import pandas as pd

def run_evaluation(questions, documents, metrics=RAGAS_METRICS) -> pd.DataFrame:
    examples = []

    # Build the retriever and LangGraph app ONCE
    app, retriever = baseline_rag_app(documents)

    # Run the app for each question
    for q in questions:
        result = app.invoke({
            "question": q,
            "retriever": retriever
        })
        examples.append({
            "query": q,
            "contexts": result["retrieved_docs"],
            "ground_truth": "",  # Leave empty for now
            "generation": result["answer"],
        })

        dataset = Dataset.from_dict({
            "question": [e["query"] for e in examples],
            "contexts": [e["contexts"] for e in examples],
            "ground_truths": [e["ground_truth"] for e in examples],
            "response": [e["generation"] for e in examples],
            "reference": [e["ground_truth"] for e in examples],
        })


    evaluation = evaluate(dataset, metrics=metrics)
    return evaluation.to_pandas()
