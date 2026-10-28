import json
import re
from pathlib import Path
from src.rag import answer_basic, answer_corrective

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "evaluation_questions.json"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def source_hit(result, expected_sources):
    found = {item["source"] for item in result.get("retrieved", [])}
    return not expected_sources or bool(found.intersection(expected_sources))


def run(mode="basic"):
    questions = json.loads(DATASET.read_text())
    correct = 0
    source_correct = 0
    rows = []

    for item in questions:
        result = answer_basic(item["question"]) if mode == "basic" else answer_corrective(item["question"])
        answer = result["answer"]
        # This is deliberately a lightweight baseline metric. We will replace it with
        # LLM-as-judge / LangSmith evaluation once the first run works.
        expected_terms = [t for t in normalize(item["expected_answer"]).split() if len(t) > 5]
        overlap = sum(term in normalize(answer) for term in expected_terms) / max(len(expected_terms), 1)
        is_correct = overlap >= 0.35 if item["expected_sources"] else "couldn t find sufficient" in normalize(answer)
        is_source = source_hit(result, item["expected_sources"])
        correct += int(is_correct)
        source_correct += int(is_source)
        rows.append({"id": item["id"], "correct": is_correct, "source_hit": is_source, "answer": answer})
        print(f"{item['id']}: correct={is_correct}, source_hit={is_source}")

    print(f"\nMode: {mode}")
    print(f"Answer score: {correct}/{len(questions)} = {correct/len(questions):.1%}")
    print(f"Source retrieval: {source_correct}/{len(questions)} = {source_correct/len(questions):.1%}")
    return rows


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "basic")
