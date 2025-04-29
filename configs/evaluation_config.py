# configs/evaluation_config.py

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

# List of metrics to evaluate with
RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
]
