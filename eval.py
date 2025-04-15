import asyncio
import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from grader import grade_sample
from linkup import LinkupClient
from tavily import TavilyClient
from tqdm import tqdm

load_dotenv()

linkup_api_key = os.getenv("LINKUP_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


def get_data():
    df = pd.read_csv("simple_qa_test_set.csv")
    # Add filtering for valid rows if needed
    return df


def sample_questions(n: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = get_data()
    return df.sample(n=n, random_state=seed) if n is not None else df


# Concurrency settings
MAX_CONCURRENT_TASKS = 5  # Max concurrent questions per policy
MAX_CONCURRENT_POLICIES = 2  # Max concurrent policies

# Create a thread-safe queue for logging
log_queue = queue.Queue()
log_lock = threading.Lock()


def print_log(policy_type: str, message: str):
    """Thread-safe logging with proper formatting."""
    with log_lock:
        print(f"[{policy_type}] {message}")


async def run_linkup_policy(
    question: str,
    policy_args: dict[str, Any] | None,
) -> Tuple[str, None]:
    """Run linkup policy in a thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: LinkupClient(api_key=linkup_api_key, **policy_args or dict())
            .search(question, depth="deep", output_type="sourcedAnswer")
            .answer,
        )
        return result, None


async def run_linkup_standard_policy(
    question: str,
    policy_args: dict[str, Any] | None,
) -> Tuple[str, None]:
    """Run linkup policy in a thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: LinkupClient(api_key=linkup_api_key, **policy_args or dict())
            .search(question, depth="standard", output_type="sourcedAnswer")
            .answer,
        )
        return result, None


async def run_tavily_policy(
    question: str,
    policy_args: dict[str, Any] | None,
) -> Tuple[str, None]:
    """Run tavily policy in a thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: TavilyClient(api_key=tavily_api_key, **policy_args or dict()).search(
                question, search_depth="advanced", include_answer=True
            )["answer"],
        )
        return result, None


async def run_policy_async(
    question: str,
    policy_type: str = "linkup",
    policy_args: dict[str, Any] | None = None,
) -> Tuple[str, Optional[Any]]:
    """Async version of run_policy."""
    policy_handlers = {
        "tavily": run_tavily_policy,
        "linkup": run_linkup_policy,
        "linkup_standard": run_linkup_standard_policy,
    }
    if policy_type not in policy_handlers:
        raise ValueError(f"Unknown policy type: {policy_type}")

    return await policy_handlers[policy_type](question=question, policy_args=policy_args)


def save_trace(state, grade_letter, timestamp, base_dir="traces"):
    """Save policy traces to appropriate directory based on grade."""
    if state is None:
        return

    trace_dir = os.path.join(base_dir, "correct" if grade_letter == "A" else "incorrect")
    os.makedirs(trace_dir, exist_ok=True)
    state.save(os.path.join(trace_dir, f"trace_{timestamp}.txt"))


def calculate_f_score(metrics: Dict[str, float]) -> float:
    """Calculate F score from metrics.

    Args:
        metrics: Dictionary containing accuracy_given_attempted and is_correct

    Returns:
        float: F-score
    """
    if (metrics["accuracy_given_attempted"] + metrics["is_correct"]) > 0:
        return (
            2
            * metrics["accuracy_given_attempted"]
            * metrics["is_correct"]
            / (metrics["accuracy_given_attempted"] + metrics["is_correct"])
        )
    return 0.0


def calculate_metrics(results: list) -> Dict[str, float]:
    """Calculate aggregate metrics from results."""
    total = len(results)
    if not total:
        return {
            "is_correct": 0,
            "is_incorrect": 0,
            "is_not_attempted": 0,
            "is_given_attempted": 0,
            "accuracy_given_attempted": 0,
            "avg_latency": 0,
        }

    counts = {"A": 0, "B": 0, "C": 0}
    latencies = []

    for grade, latency in results:
        counts[grade] = counts.get(grade, 0) + 1
        latencies.append(latency)

    metrics = {
        "is_correct": counts["A"] / total,
        "is_incorrect": counts["B"] / total,
        "is_not_attempted": counts["C"] / total,
        "avg_latency": sum(latencies) / len(latencies),
    }

    metrics["is_given_attempted"] = metrics["is_correct"] + metrics["is_incorrect"]
    metrics["accuracy_given_attempted"] = (
        metrics["is_correct"] / metrics["is_given_attempted"]
        if metrics["is_given_attempted"] > 0
        else 0
    )

    metrics["f_score"] = calculate_f_score(metrics)
    return metrics


async def run_evaluation(policy_type: str, questions_df: pd.DataFrame) -> Dict[str, float]:
    """Run evaluation for a single policy type.

    Args:
        policy_type: Type of policy to use
        questions_df: DataFrame containing the questions to evaluate
    """
    os.makedirs("traces", exist_ok=True)
    results = []

    print(f"\nEvaluating {policy_type.upper()} policy on {len(questions_df)} samples...")
    print("-" * 100)

    for problem, answer in zip(questions_df["problem"], questions_df["answer"]):
        start_time = datetime.now()
        predicted_answer, policy_state = await run_policy_async(problem, policy_type)
        latency = (datetime.now() - start_time).total_seconds()

        print(f"Question: {problem}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct Answer: {answer}")
        print(f"Latency: {latency:.2f} seconds")

        grade_letter = grade_sample(problem, answer, predicted_answer)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_trace(policy_state, grade_letter, timestamp)

        results.append((grade_letter, latency))
        print(f"Grade: {grade_letter}")
        print("-" * 100)

    metrics = calculate_metrics(results)
    return metrics


def print_metrics(metrics: Dict[str, float], policy_name: str | None = None):
    """Print metrics in a formatted way."""
    if policy_name:
        print(f"\nMETRICS FOR {policy_name.upper()}")
    print("##################")
    print(f"Accuracy Given Attempted: {metrics['accuracy_given_attempted']:.3f}")
    print(f"F Score: {metrics['f_score']:.3f}")
    print(f"Average Latency: {metrics['avg_latency']:.2f} seconds")
    print(f"Correct: {metrics['is_correct']:.3f}")
    print(f"Incorrect: {metrics['is_incorrect']:.3f}")
    print(f"Not Attempted: {metrics['is_not_attempted']:.3f}")


async def compare_policies(policy1: str, policy2: str, num_samples: int):
    """Compare two policies on the same set of questions."""
    print(f"\nComparing {policy1.upper()} vs {policy2.upper()} on {num_samples} samples...")

    # Use the same samples for both policies
    questions_df = sample_questions(num_samples)

    metrics1 = await run_evaluation(policy1, questions_df)
    metrics2 = await run_evaluation(policy2, questions_df)

    print("\nCOMPARISON RESULTS")
    print("=" * 50)
    print_metrics(metrics1, policy1)
    print("\n" + "-" * 30)
    print_metrics(metrics2, policy2)


def generate_question_id(question: str) -> str:
    """Generate a unique, deterministic ID for a question."""
    return hashlib.sha256(question.encode()).hexdigest()[
        :16
    ]  # First 16 chars of hash is sufficient


async def evaluate_questions_async(
    questions_df: pd.DataFrame,
    policy_type: str,
    policy_args: dict[str, Any] | None,
) -> list:
    """Evaluate questions and return results."""
    sem = Semaphore(MAX_CONCURRENT_TASKS)
    results = []
    result_manager = ResultManager(policy_type)

    # Load existing results if any
    if result_manager.results_file.exists():
        with open(result_manager.results_file, "r") as f:
            existing_results = [json.loads(line) for line in f]
            results.extend((r["grade"], r["latency"]) for r in existing_results)
            print(f"\nLoaded {len(existing_results)} existing results for {policy_type} policy")

    async def process_question(q_id: str, problem: str, answer: str) -> Optional[Tuple[str, float]]:
        # Skip if already processed for this specific policy
        cache_key = f"{q_id}_{policy_type}"  # Combine question ID and policy type
        if cache_key in result_manager.processed_indices:
            return None

        async with sem:
            retries = 3
            for attempt in range(retries):
                start_time = time.time()
                try:
                    predicted_answer, _ = await asyncio.wait_for(
                        run_policy_async(
                            question=problem,
                            policy_type=policy_type,
                            policy_args=policy_args,
                        ),
                        timeout=300,
                    )
                    latency = time.time() - start_time
                    grade_letter = grade_sample(problem, answer, predicted_answer)

                    result = {
                        "question_id": q_id,
                        "policy_type": policy_type,
                        "grade": grade_letter,
                        "question": problem,
                        "predicted": predicted_answer,
                        "correct": answer,
                        "latency": latency,
                        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "attempt": attempt + 1,
                    }
                    result_manager.save_result(result)
                    result_manager.save_checkpoint(cache_key)  # Save with combined key

                    return (grade_letter, latency)

                except asyncio.TimeoutError:
                    print(f"Timeout on question {q_id}, attempt {attempt + 1}/{retries}")
                    if attempt == retries - 1:
                        error_msg = "Maximum retries exceeded due to timeouts"
                        break
                    await asyncio.sleep(5 * (attempt + 1))  # Exponential backoff

                except Exception as e:
                    print(f"Error on question {q_id}, attempt {attempt + 1}/{retries}: {str(e)}")
                    if attempt == retries - 1:
                        error_msg = str(e)
                        break
                    await asyncio.sleep(5 * (attempt + 1))

            # If all retries failed, save error result
            error_result = {
                "question_id": q_id,
                "policy_type": policy_type,
                "grade": "C",
                "question": problem,
                "predicted": f"ERROR: {error_msg}",
                "correct": answer,
                "latency": time.time() - start_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "error": error_msg,
                "attempts": retries,
            }
            result_manager.save_result(error_result)
            result_manager.save_checkpoint(cache_key)
            return ("C", 0.0)

    # Filter out already processed questions using combined keys
    question_ids = [generate_question_id(q) for q in questions_df["problem"]]
    remaining_questions = [
        (q_id, problem, answer)
        for q_id, problem, answer in zip(
            question_ids, questions_df["problem"], questions_df["answer"]
        )
        if f"{q_id}_{policy_type}" not in result_manager.processed_indices
    ]

    print(f"\nTotal questions: {len(questions_df)}")
    print(f"Already processed: {len(questions_df) - len(remaining_questions)}")
    print(f"Remaining to process: {len(remaining_questions)}")

    tasks = [
        process_question(q_id, problem, answer) for q_id, problem, answer in remaining_questions
    ]

    if tasks:  # Only process if there are remaining tasks
        with tqdm(total=len(tasks), desc=f"Evaluating {policy_type}") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:  # Skip None results (already processed)
                    grade, latency = result
                    results.append(result)
                    pbar.set_postfix({"Last Grade": grade, "Latency": f"{latency:.2f}s"})
                pbar.update(1)

                # Periodically show summary and update summary file (every 10 questions)
                if pbar.n % 10 == 0 and pbar.n > 0:
                    current_metrics = calculate_metrics(results)
                    print(f"\nInterim metrics after {pbar.n} questions:")
                    print(f"Accuracy: {current_metrics['accuracy_given_attempted']:.3f}")
                    print(f"Average latency: {current_metrics['avg_latency']:.2f}s")
                    # Update summary file with current metrics
                    result_manager.save_summary(current_metrics)

    # Save summary metrics using all results (existing + new)
    metrics = calculate_metrics(results)
    result_manager.save_summary(metrics)

    return results


def analyze_results(results_file: Path):
    """Analyze saved results."""
    df = ResultManager.load_results(results_file)

    # Basic analysis
    grade_counts = df["grade"].value_counts()
    avg_latency = df["latency"].mean()

    print("\nResults Analysis")
    print("=" * 50)
    print(f"Total questions: {len(df)}")
    print("\nGrade Distribution:")
    for grade, count in grade_counts.items():
        print(f"Grade {grade}: {count} ({count / len(df) * 100:.2f}%)")
    print(f"\nAverage latency: {avg_latency:.2f}s")

    return df


async def compare_policies_async(
    policy1: str,
    policy1_args: dict[str, Any] | None,
    policy2: str,
    policy2_args: dict[str, Any] | None,
    num_samples: int,
    seed: int,
) -> None:
    """Compare two policies on the same set of questions."""
    questions_df = sample_questions(n=num_samples, seed=seed)

    print(f"\nEvaluating {policy1}...")
    results1 = await evaluate_questions_async(
        questions_df=questions_df,
        policy_type=policy1,
        policy_args=policy1_args,
    )

    print(f"\nEvaluating {policy2}...")
    results2 = await evaluate_questions_async(
        questions_df=questions_df,
        policy_type=policy2,
        policy_args=policy2_args,
    )

    metrics1 = calculate_metrics(results1)
    metrics2 = calculate_metrics(results2)

    print("\nCOMPARISON RESULTS")
    print("=" * 50)
    print_metrics(metrics1, policy1)
    print("\n" + "-" * 30)
    print_metrics(metrics2, policy2)


class ResultManager:
    """Manages saving and loading of evaluation results."""

    def __init__(self, policy_type: str):
        self.policy_type = policy_type
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Use policy type only for file names, without timestamp
        self.results_file = self.results_dir / f"{policy_type}_results.jsonl"
        self.summary_file = self.results_dir / f"{policy_type}_summary.json"
        self.checkpoint_file = self.results_dir / f"{policy_type}_checkpoint.json"

        # Track processed indices
        self.processed_indices = self.load_checkpoint()

    def load_checkpoint(self) -> set:
        """Load checkpoint of processed indices if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_checkpoint(self, cache_key: str):
        """Save checkpoint of processed cache_key."""
        self.processed_indices.add(cache_key)
        with open(self.checkpoint_file, "w") as f:
            json.dump(list(self.processed_indices), f)

    def save_result(self, result: dict):
        """Save individual result in JSONL format."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def save_summary(self, metrics: dict):
        """Save summary metrics."""
        with open(self.summary_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and compare QA policies")
    parser.add_argument(
        "--mode",
        choices=["evaluate", "compare"],
        required=True,
        help="Mode of operation: evaluate a single policy or compare two policies",
    )
    parser.add_argument(
        "--policy1",
        choices=["linkup", "linkup_standard", "tavily"],
        help="First (or only) policy to evaluate",
    )
    parser.add_argument(
        "--policy1-args",
        type=str,
        default="{}",
        help="Additional arguments for the first policy in JSON format",
    )
    parser.add_argument(
        "--policy2",
        choices=["linkup", "linkup_standard", "tavily"],
        help="Second policy to compare against (only in compare mode)",
    )
    parser.add_argument(
        "--policy2-args",
        type=str,
        default="{}",
        help="Additional arguments for the second policy in JSON format",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate (if not specified, uses complete dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling questions",
    )
    parser.add_argument("--analyze", type=str, help="Path to results file to analyze")

    args = parser.parse_args()

    async def main():
        try:
            if args.mode == "evaluate":
                if not args.policy1:
                    parser.error("--policy1 is required for evaluate mode")
                questions_df = sample_questions(n=args.num_samples, seed=args.seed)
                results = await evaluate_questions_async(
                    questions_df=questions_df,
                    policy_type=args.policy1,
                    policy_args=json.loads(args.policy1_args),
                )
                metrics = calculate_metrics(results)
                print_metrics(metrics, args.policy1)
            else:  # compare mode
                if not args.policy1 or not args.policy2:
                    parser.error("Both --policy1 and --policy2 are required for compare mode")
                await compare_policies_async(
                    policy1=args.policy1,
                    policy1_args=json.loads(args.policy1_args),
                    policy2=args.policy2,
                    policy2_args=json.loads(args.policy2_args),
                    num_samples=args.num_samples,
                    seed=args.seed,
                )
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Progress has been saved.")
            sys.exit(1)
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    if args.analyze:
        # Analyze existing results
        analyze_results(Path(args.analyze))
    else:
        # Run evaluation
        asyncio.run(main())
