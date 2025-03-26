# SimpleQA Evaluator

A tool for evaluating and comparing different Question Answering (QA) policies on OpenAI's SimpleQA benchmark, measuring metrics like f-score, accuracy, and latency.

## Features

The evaluator provides comprehensive testing capabilities for AI search engines including Linkup Deep, Linkup Standard, and Tavily APIs. It supports both single-policy evaluation and head-to-head comparisons, with built-in async processing and progress tracking.

## Setup

1. Install dependencies:

```bash
pip install pandas tqdm python-dotenv linkup-sdk tavily-python openai
```

2. Create a `.env` file with your API keys:
```
LINKUP_API_KEY=your_linkup_key
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
```

3. Ensure you have the `simple_qa_test_set.csv` file containing the SimpleQA benchmark in your project directory.

## Usage

### Evaluate a Single Policy
```bash
python eval.py --mode evaluate --policy1 linkup --num-samples 100
```

### Compare Two Policies
```bash
python eval.py --mode compare --policy1 linkup --policy2 tavily --num-samples 100
```

### Analyze Existing Results
```bash
python eval.py --analyze results/linkup_results.jsonl
```

## Supported QA Policies

The evaluator currently supports three QA policies:

- Linkup API (deep search mode)
- Linkup API (standard search mode)
- Tavily API (advanced search mode)

## Output and Metrics

The evaluator generates comprehensive results in the `results` directory, including detailed per-question analysis and summary metrics. Key performance indicators include accuracy on attempted questions, F-score, average latency, and attempt rates.

Results are automatically saved as:

- Detailed results: `{policy}_results.jsonl`
- Summary metrics: `{policy}_summary.json`
- Progress tracking: `{policy}_checkpoint.json`

## Reliability Features

The evaluator includes robust error handling with automatic retries for failed requests, 5-minute timeouts per question, and checkpoint saving to resume interrupted evaluations.

## Benchmark Results

Here's a comparison of different QA policies on the SimpleQA benchmark:

| Policy | F-Score  | Accuracy | Attempt Rate | Avg Latency (s) |
|--------|----------|----------|--------------|-----------------|
| Linkup (Deep) | 0.910 | 8.23 | 99.5% | 8.23 |
| Linkup (Standard) | 0.850 | 0.837 | 96.8% | 4.10 |
| Tavily | 0.728 | 0.726 | 99.6% | 2.92 |

As shown in the results:

- Linkup Deep Search achieves the highest f-score and accuracy, with excellent attempt rates
- Linkup Standard offers a good balance of performance and speed
- Tavily provides the fastest responses while maintaining high attempt rates
