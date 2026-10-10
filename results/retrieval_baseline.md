\# MedAssist-AI Retrieval Benchmark — Baseline



\## Evaluation Setup



\- Evaluation questions: 3

\- Retrieved candidates per question: Top 10

\- Metrics: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and latency



\## Results



| K | Precision@K | Recall@K | MRR |

|---|---:|---:|---:|

| 3 | 0.4444 | 0.7222 | 0.6111 |

| 5 | 0.3333 | 0.8333 | 0.6111 |

| 10 | 0.1667 | 0.8333 | 0.6111 |



Mean reported retrieval latency: 1330.41 ms.



\## Observations



\- Precision decreases as the number of retrieved chunks increases.

\- Recall improves from K=3 to K=5 and remains unchanged at K=10.

\- The evaluation dataset contains only three questions.

\- Relevance labels need further verification before drawing strong conclusions.



\## Next Steps



1\. Expand the evaluation dataset.

2\. Verify relevant document IDs against their actual content.

3\. Investigate retrieval ranking and irrelevant chunks.

4\. Compare future results against this baseline.

