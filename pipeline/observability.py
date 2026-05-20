import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from pipeline.llm import LLMClient


class ObservabilityClient(LLMClient):
    """
    Wraps any LLMClient and logs every call to a local JSONL trace file.
    Optionally forwards to LangSmith when LANGSMITH_API_KEY is set.

    Usage:
        llm = ObservabilityClient(
            client=GroqClient(),
            run_name="extraction",
            trace_path="traces/extraction.jsonl",
        )

    LangSmith (optional):
        Set LANGSMITH_API_KEY in .env and pip install langsmith.
    """

    def __init__(
        self,
        client: LLMClient,
        run_name: str = "pipeline",
        trace_path: str = "traces.jsonl",
    ):
        self.client = client
        self.run_name = run_name
        self.trace_path = trace_path
        self._langsmith = self._init_langsmith()

    def _init_langsmith(self):
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            return None
        try:
            from langsmith import Client
            client = Client(api_key=api_key)
            project = os.getenv("LANGSMITH_PROJECT", "pipeline")
            print(f"  LangSmith tracing enabled — project: {project}")
            return {"client": client, "project": project}
        except ImportError:
            print("  LANGSMITH_API_KEY set but langsmith not installed. Run: pip install langsmith")
            return None

    def get_response(self, messages: list) -> str:
        start = time.time()
        response: Optional[str] = None
        error: Optional[str] = None

        try:
            response = self.client.get_response(messages)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = int((time.time() - start) * 1000)
            self._record(messages, response, latency_ms, error)

    def _record(self, messages: list, response: Optional[str], latency_ms: int, error: Optional[str]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_name": self.run_name,
            "model": getattr(self.client, "last_model_used", None),
            "latency_ms": latency_ms,
            "succeeded": error is None,
            "error": error,
            "n_messages": len(messages),
            "input_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "output_chars": len(response) if response else 0,
            "output_preview": response[:300].replace("\n", " ") if response else None,
        }

        os.makedirs(os.path.dirname(self.trace_path) if os.path.dirname(self.trace_path) else ".", exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self._langsmith:
            self._send_to_langsmith(messages, response, latency_ms, error, record)

    def _send_to_langsmith(self, messages, response, latency_ms, error, record):
        try:
            from langsmith.schemas import RunTypeEnum
            import uuid

            client = self._langsmith["client"]
            project = self._langsmith["project"]
            run_id = str(uuid.uuid4())
            start_time = datetime.now(timezone.utc)

            client.create_run(
                id=run_id,
                name=self.run_name,
                run_type=RunTypeEnum.llm,
                project_name=project,
                inputs={"messages": messages},
                outputs={"response": response} if response else None,
                error=error,
                start_time=start_time,
                extra={"model": record.get("model"), "latency_ms": latency_ms},
            )
        except Exception as e:
            print(f"  LangSmith log failed (non-fatal): {e}")


def load_traces(trace_path: str = "traces.jsonl") -> list[dict]:
    """Load trace records from a JSONL file for analysis."""
    if not os.path.exists(trace_path):
        return []
    with open(trace_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trace_summary(trace_path: str = "traces.jsonl") -> None:
    """Print a quick summary of a trace file."""
    records = load_traces(trace_path)
    if not records:
        print(f"No traces found at {trace_path}.")
        return

    total = len(records)
    failed = sum(1 for r in records if not r["succeeded"])
    latencies = [r["latency_ms"] for r in records if r["succeeded"]]
    avg_ms = int(sum(latencies) / len(latencies)) if latencies else 0
    p95_ms = int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0

    models = {}
    for r in records:
        m = r.get("model") or "unknown"
        models[m] = models.get(m, 0) + 1

    print(f"\n=== Trace summary: {trace_path} ===")
    print(f"Total calls   : {total}")
    print(f"Failures      : {failed} ({failed/total:.1%})")
    print(f"Avg latency   : {avg_ms}ms")
    print(f"p95 latency   : {p95_ms}ms")
    print(f"\nModel breakdown:")
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
        print(f"  {model:<45} {count:>4} calls ({count/total:.1%})")
