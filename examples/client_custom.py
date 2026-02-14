#!/usr/bin/env python3

import argparse
import time
import requests
import statistics
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2


# ---------------- GPU MONITOR ---------------- #

class GPUMonitor:
    def __init__(self):
        self.samples = []
        self.running = False

    def _collect(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"]
                ).decode().strip()
                util, mem = out.split(",")
                self.samples.append((float(util), float(mem)))
            except Exception:
                pass
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def summary(self):
        if not self.samples:
            return {}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "gpu_mean": statistics.mean(utils),
            "gpu_p90": statistics.quantiles(utils, n=100)[89] if len(utils) > 1 else utils[0],
            "mem_peak": max(mems),
        }


# ---------------- REQUEST WORKER ---------------- #

def request_one(session, url, payload, idx):
    t0 = time.perf_counter()
    try:
        with session.post(
            url,
            json=payload,
            stream=True,
            timeout=300,
            headers={"Accept": "audio/L16"}
        ) as resp:

            resp.raise_for_status()
            chunks = []
            first_latency = None

            for chunk in resp.iter_content(4096):
                if not chunk:
                    continue
                if first_latency is None:
                    first_latency = time.perf_counter() - t0
                chunks.append(chunk)

            pcm = b"".join(chunks)

        total_time = time.perf_counter() - t0
        duration = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
        rtf = total_time / duration if duration > 0 else 0

        return {
            "ok": True,
            "idx": idx,
            "type": payload["type"],
            "first_latency": first_latency,
            "total_time": total_time,
            "duration": duration,
            "rtf": rtf,
        }

    except Exception as e:
        return {"ok": False, "idx": idx, "err": str(e)}


# ---------------- MAIN ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--small", type=int, default=4)
    p.add_argument("--long", type=int, default=4)
    args = p.parse_args()

    url = f"{args.base_url.rstrip('/')}/v1/audio/speech"

    short_text = "Hi, this is a test and my name is sahil"
    long_text = " ".join(["This is a long synthesis stress test sentence."] * 2)

    workload = []
    idx = 0

    for _ in range(args.small):
        workload.append({
            "idx": idx,
            "payload": {
                "text": short_text,
                "language": "English",
                "speaker": "Vivian",
                "type": "small"
            }
        })
        idx += 1

    for _ in range(args.long):
        workload.append({
            "idx": idx,
            "payload": {
                "text": long_text,
                "language": "English",
                "speaker": "Vivian",
                "type": "long"
            }
        })
        idx += 1

    print(f"\nTotal requests: {len(workload)}")
    print(f"Parallel workers: {args.parallel}\n")

    session = requests.Session()

    monitor = GPUMonitor()
    monitor.start()

    start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(request_one, session, url, w["payload"], w["idx"]): w["idx"]
            for w in workload
        }

        for f in as_completed(futures):
            r = f.result()
            results.append(r)

            if r["ok"]:
                print(
                    f"[DONE] idx={r['idx']} "
                    f"type={r['type']} "
                    f"first={r['first_latency']:.2f}s "
                    f"total={r['total_time']:.2f}s "
                    f"rtf={r['rtf']:.2f}"
                )
            else:
                print(f"[FAIL] idx={r['idx']} err={r['err']}")

    total_wall = time.perf_counter() - start
    monitor.stop()

    small = [r for r in results if r.get("type") == "small" and r["ok"]]
    long = [r for r in results if r.get("type") == "long" and r["ok"]]

    def summarize(arr):
        if not arr:
            return {}
        return {
            "count": len(arr),
            "first_mean": statistics.mean([x["first_latency"] for x in arr]),
            "total_mean": statistics.mean([x["total_time"] for x in arr]),
            "rtf_mean": statistics.mean([x["rtf"] for x in arr]),
        }

    print("\n=== SUMMARY ===")
    print("Wall time:", round(total_wall, 2), "s")
    print("Throughput:", round(len(results) / total_wall, 2), "req/s")

    print("\n--- SMALL ---")
    print(summarize(small))

    print("\n--- LONG ---")
    print(summarize(long))

    print("\n--- GPU ---")
    print(monitor.summary())


if __name__ == "__main__":
    main()
