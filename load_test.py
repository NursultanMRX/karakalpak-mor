"""
Load test for Karakalpak POS+Morph API.

Tests concurrent users, response times, rate limiting, and resource behavior.

Usage:
  pip install httpx psutil
  python load_test.py --url http://localhost:8000 --key kaa_YOUR_KEY
  python load_test.py --url https://api.kkgrammar.uz --key kaa_YOUR_KEY
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import httpx
except ImportError:
    print("ERROR: Run: pip install httpx")
    sys.exit(1)

# ── Test sentences (Karakalpak) ──────────────────────────────────────────────
SENTENCES = [
    "Men mektepke baraman",
    "Ol kitap oqıydı",
    "Biz universitet studentlerimiz",
    "Siz qaydan keldiñiz",
    "Olar shaarda jasaydı",
    "Meniñ atam diyqan",
    "Qala ishinde ko'p adam bar",
    "Bul jaqsi kitap edi",
    "Men karakalpaq tilinde so'yleyman",
    "Bizdiñ el bayrak astında birlesken",
]

COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

def c(color, text):
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"


@dataclass
class Result:
    user_id: int
    request_id: int
    status: int
    latency_ms: float
    error: Optional[str] = None
    rate_limited: bool = False
    queued: bool = False


@dataclass
class ScenarioReport:
    concurrent_users: int
    total_requests: int
    results: List[Result] = field(default_factory=list)

    @property
    def successes(self):
        return [r for r in self.results if r.status == 200]

    @property
    def rate_limited(self):
        return [r for r in self.results if r.rate_limited]

    @property
    def errors(self):
        return [r for r in self.results if r.status not in (200, 429) and r.error]

    @property
    def success_rate(self):
        return len(self.successes) / len(self.results) * 100 if self.results else 0

    @property
    def latencies(self):
        return [r.latency_ms for r in self.successes]

    @property
    def p50(self):
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self):
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    @property
    def p99(self):
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.99)]

    @property
    def avg(self):
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def throughput(self):
        if not self.results:
            return 0
        total_time = max(r.latency_ms for r in self.results) / 1000
        return len(self.successes) / total_time if total_time > 0 else 0


async def single_request(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    user_id: int,
    request_id: int,
    sentence: str,
) -> Result:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key

    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/predict",
            json={"sentence": sentence},
            headers=headers,
            timeout=60.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        rate_limited = resp.status_code == 429
        return Result(
            user_id=user_id,
            request_id=request_id,
            status=resp.status_code,
            latency_ms=latency_ms,
            rate_limited=rate_limited,
        )
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000
        return Result(user_id=user_id, request_id=request_id, status=0,
                      latency_ms=latency_ms, error="TIMEOUT")
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return Result(user_id=user_id, request_id=request_id, status=0,
                      latency_ms=latency_ms, error=str(e))


async def run_user(
    url: str, key: str, user_id: int, requests_per_user: int, results: list
):
    async with httpx.AsyncClient() as client:
        for i in range(requests_per_user):
            sentence = SENTENCES[(user_id * requests_per_user + i) % len(SENTENCES)]
            result = await single_request(client, url, key, user_id, i, sentence)
            results.append(result)
            status_str = (
                c("green", f"200 ({result.latency_ms:.0f}ms)")
                if result.status == 200
                else c("yellow", f"429 RATE LIMITED")
                if result.rate_limited
                else c("red", f"{result.status} {result.error or ''}")
            )
            print(f"  User{user_id+1} req{i+1}: {status_str}")


async def run_scenario(
    url: str, key: str, concurrent_users: int, requests_per_user: int
) -> ScenarioReport:
    report = ScenarioReport(
        concurrent_users=concurrent_users,
        total_requests=concurrent_users * requests_per_user,
    )
    results = []

    print(c("cyan", f"\n{'─'*60}"))
    print(c("bold", f"  Scenario: {concurrent_users} concurrent user(s), {requests_per_user} request(s) each"))
    print(c("cyan", f"{'─'*60}"))

    start = time.perf_counter()
    tasks = [
        run_user(url, key, uid, requests_per_user, results)
        for uid in range(concurrent_users)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    report.results = results
    return report, elapsed


def print_report(report: ScenarioReport, elapsed: float):
    print()
    ok = c("green", "✅") if report.success_rate >= 90 else c("yellow", "⚠️") if report.success_rate >= 50 else c("red", "❌")
    print(f"  {ok} Success rate : {report.success_rate:.1f}%  ({len(report.successes)}/{len(report.results)})")

    if report.latencies:
        print(f"  ⏱  Latency      : avg={report.avg:.0f}ms  p50={report.p50:.0f}ms  p95={report.p95:.0f}ms  p99={report.p99:.0f}ms")
        print(f"  🚀 Throughput   : {report.throughput:.2f} req/s")

    if report.rate_limited:
        print(c("yellow", f"  ⚡ Rate limited : {len(report.rate_limited)} requests got 429 (expected — per-IP limit)"))

    if report.errors:
        print(c("red", f"  💥 Errors       : {len(report.errors)} requests failed"))
        for r in report.errors[:3]:
            print(c("red", f"     User{r.user_id+1} req{r.request_id+1}: {r.error}"))

    print(f"  🕐 Total time   : {elapsed:.1f}s")


def print_final_summary(all_reports):
    print(c("bold", f"\n{'═'*60}"))
    print(c("bold", "  FINAL REPORT"))
    print(c("bold", f"{'═'*60}"))
    print(f"  {'Users':>6}  {'Success':>8}  {'Avg ms':>8}  {'p95 ms':>8}  {'429s':>6}  {'Errors':>6}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for report, elapsed in all_reports:
        sr = f"{report.success_rate:.0f}%"
        avg = f"{report.avg:.0f}" if report.latencies else "—"
        p95 = f"{report.p95:.0f}" if report.latencies else "—"
        rl = str(len(report.rate_limited))
        err = str(len(report.errors))
        color = "green" if report.success_rate >= 90 else "yellow" if report.success_rate >= 50 else "red"
        print(f"  {report.concurrent_users:>6}  {c(color, sr):>18}  {avg:>8}  {p95:>8}  {rl:>6}  {err:>6}")

    print(f"\n  Legend:")
    print(f"  {c('green','✅ ≥90% success')}  {c('yellow','⚠️  50-90%')}  {c('red','❌ <50%')}")
    print(f"  429 = Rate limited (safe, expected behavior)")
    print(f"  Errors = Timeout or crash (bad)")
    print(c("bold", f"{'═'*60}\n"))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key", default="", help="X-API-Key value")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer requests)")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    print(c("bold", f"\n{'═'*60}"))
    print(c("bold", "  Karakalpak API — Load Test"))
    print(c("bold", f"{'═'*60}"))
    print(f"  Target : {url}")
    print(f"  Auth   : {'✅ API key set' if args.key else '⚠️  No API key'}")

    # Health check first
    print(f"\n  Checking /health...")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                print(c("green", f"  ✅ API healthy — model_loaded={data.get('model_loaded')} quantized={data.get('quantized')}"))
            else:
                print(c("red", f"  ❌ Health check failed: {r.status_code}"))
                sys.exit(1)
    except Exception as e:
        print(c("red", f"  ❌ Cannot reach API: {e}"))
        sys.exit(1)

    # Scenarios: (concurrent_users, requests_per_user)
    if args.quick:
        scenarios = [(1, 2), (2, 2), (5, 1)]
    else:
        scenarios = [(1, 3), (2, 3), (5, 2), (10, 2), (20, 1)]

    all_reports = []
    for users, reqs in scenarios:
        report, elapsed = await run_scenario(url, args.key, users, reqs)
        print_report(report, elapsed)
        all_reports.append((report, elapsed))
        await asyncio.sleep(2)  # brief pause between scenarios

    print_final_summary(all_reports)


if __name__ == "__main__":
    asyncio.run(main())
