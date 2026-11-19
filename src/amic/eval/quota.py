"""Run judge jobs against a daily-capped API (Gemini free tier: 500 requests/day).

Jobs run in the given (priority) order. When the provider reports its DAILY cap, workers stop
taking new jobs; with `wait=True` the runner sleeps until the reported reset and resumes with
the jobs that did not finish. Results are handed to `sink` as they complete (callers append
them to a resumable JSONL), so an interrupted run loses nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, TypeVar

from amic.generation.gateway.openai_compat import DailyQuotaExceeded

J = TypeVar("J")


def run_with_quota(
    jobs: Sequence[J],
    fn: Callable[[J], Any],
    sink: Callable[[Any], None],
    *,
    workers: int = 4,
    wait: bool = False,
    label: str = "jobs",
    margin_s: float = 180.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Returns the number of jobs completed. Failures other than the quota are printed."""
    pending = list(jobs)
    done_total = 0
    while pending:
        stop = threading.Event()
        quota: list[DailyQuotaExceeded] = []
        finished: set[int] = set()

        def guarded(
            i: int,
            job: J,
            stop: threading.Event = stop,
            quota: list[DailyQuotaExceeded] = quota,
        ) -> tuple[int, Any]:
            if stop.is_set():
                raise _Skipped
            try:
                return i, fn(job)
            except DailyQuotaExceeded as exc:
                stop.set()
                quota.append(exc)
                raise _Skipped from exc

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(guarded, i, j) for i, j in enumerate(pending)]
            for fut in as_completed(futs):
                try:
                    i, res = fut.result()
                except _Skipped:
                    continue
                except Exception as err:  # noqa: BLE001 - one bad item must not stop the run
                    print(f"  FAIL {type(err).__name__}: {err}", flush=True)
                    continue
                finished.add(i)
                sink(res)
                done_total += 1
                if done_total % 25 == 0:
                    print(f"  [{label}] {done_total} done", flush=True)
        if not quota:
            break
        pending = [j for i, j in enumerate(pending) if i not in finished]
        wait_s = max(q.retry_after_s for q in quota) + margin_s
        resume = datetime.now() + timedelta(seconds=wait_s)
        print(
            f"  [{label}] daily quota exhausted after {done_total} jobs; {len(pending)} left. "
            + (f"Sleeping until {resume:%Y-%m-%d %H:%M}." if wait else "Re-run after the reset."),
            flush=True,
        )
        if not wait:
            break
        sleep(wait_s)
    print(f"  [{label}] finished: {done_total} completed this run", flush=True)
    return done_total


class _Skipped(Exception):
    pass
