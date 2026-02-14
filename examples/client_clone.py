#!/usr/bin/env python3
"""
PURE MODEL SPEED BENCHMARK CLIENT
For:
    POST /v1/audio/clone

Measures:
- First token latency
- Pure generation time (first chunk → last chunk)
- Audio duration
- Real RTF
- Throughput
- GPU utilization
"""

import asyncio
import aiohttp
import argparse
import time
import os
import statistics
import wave
import subprocess
import threading

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM


# ============================================================
# GPU MONITOR
# ============================================================

class GPUMonitor:
    def __init__(self):
        self.samples = []
        self.running = False

    def _collect(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ]
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

        utils = [x[0] for x in self.samples]
        mems = [x[1] for x in self.samples]

        return {
            "gpu_mean_%": round(statistics.mean(utils), 2),
            "gpu_p90_%": round(statistics.quantiles(utils, n=100)[89], 2) if len(utils) > 1 else utils[0],
            "mem_peak_MB": round(max(mems), 2),
        }


# ============================================================
# REQUEST WORKER
# ============================================================

async def request_one(session, url, payload, idx):
    request_start = time.perf_counter()

    first_chunk_time = None
    last_chunk_time = None
    bytes_received = 0

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/clone_req_{idx}.wav"

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
            headers={"Accept": "audio/L16"},
        ) as resp:

            if resp.status != 200:
                text = await resp.text()
                return {"ok": False, "idx": idx, "err": f"{resp.status}: {text}"}

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)

                async for chunk in resp.content.iter_chunked(4096):
                    if not chunk:
                        continue

                    now = time.perf_counter()

                    if first_chunk_time is None:
                        first_chunk_time = now

                    last_chunk_time = now
                    wf.writeframes(chunk)
                    bytes_received += len(chunk)

        if first_chunk_time is None:
            return {"ok": False, "idx": idx, "err": "No audio received"}

        # PURE MODEL GENERATION TIME
        generation_time = last_chunk_time - first_chunk_time

        # Audio duration (seconds)
        audio_duration = bytes_received / (SAMPLE_RATE * SAMPLE_WIDTH)

        # Real RTF
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        return {
            "ok": True,
            "idx": idx,
            "first_latency": first_chunk_time - request_start,
            "generation_time": generation_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
            "file": output_path,
        }

    except Exception as e:
        return {"ok": False, "idx": idx, "err": str(e)}


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--requests", type=int, default=4)

    parser.add_argument("--text", default="Hello, this is a voice clone test ok bye.")
    parser.add_argument("--ref-audio", default="/home/ubuntu/sahil/nano-qwen3tts-vllm/examples/assets/fiftyshades_anna.wav")
    parser.add_argument("--ref-text", default="I'm working at the hardware store till 7. I think I'd like that too. What?")
    parser.add_argument("--x-vector-only", action="store_true")

    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/v1/audio/clone"

    workload = []
    for i in range(args.requests):
        workload.append({
            "idx": i,
            "payload": {
                "text": args.text,
                "language": "English",
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "x_vector_only_mode": args.x_vector_only,
            },
        })

    print("\n==== PURE MODEL CLONE BENCHMARK ====")
    print(f"Total requests: {args.requests}")
    print(f"Parallel workers: {args.parallel}")
    print()

    connector = aiohttp.TCPConnector(limit=args.parallel)

    monitor = GPUMonitor()
    monitor.start()

    start_wall = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = [
            request_one(session, url, w["payload"], w["idx"])
            for w in workload
        ]

        results = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)

            if r["ok"]:
                print(
                    f"[DONE] idx={r['idx']} "
                    f"first_latency={r['first_latency']:.2f}s "
                    f"generation_time={r['generation_time']:.2f}s "
                    f"audio={r['audio_duration']:.2f}s "
                    f"rtf={r['rtf']:.2f} "
                    f"saved={r['file']}"
                )
            else:
                print(f"[FAIL] idx={r['idx']} err={r['err']}")

    total_wall = time.perf_counter() - start_wall

    monitor.stop()

    success = [r for r in results if r["ok"]]

    print("\n==== SUMMARY ====")
    print("Wall time:", round(total_wall, 2), "s")
    print("Throughput:", round(len(success) / total_wall, 2), "req/s")

    if success:
        print("Mean RTF:", round(statistics.mean([x["rtf"] for x in success]), 2))
        print("Mean Generation Time:", round(statistics.mean([x["generation_time"] for x in success]), 2), "s")
        print("Mean First Latency:", round(statistics.mean([x["first_latency"] for x in success]), 2), "s")

    print("\n==== GPU ====")
    print(monitor.summary())


if __name__ == "__main__":
    asyncio.run(main())
