"""One bounded Windows CPU baseline; standard library only."""
import ctypes as ct
import argparse
from ctypes import wintypes as wt
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
repacking = parser.add_mutually_exclusive_group()
repacking.add_argument("--no-repack", dest="no_repack", action="store_true", help="Disable CPU weight repacking (default).")
repacking.add_argument("--repack", dest="no_repack", action="store_false", help="Explicitly enable repacking for a comparison.")
parser.set_defaults(no_repack=True)
parser.add_argument("--no-warmup", action="store_true", help="Skip engine warmup for loading-stage diagnosis.")
args = parser.parse_args()
label = "olmo_no_repack_" if args.no_repack else "olmo_baseline_"
if args.no_warmup:
    label = "olmo_no_warmup_"
OUT = ROOT / "results" / (label + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUT.mkdir(parents=True, exist_ok=False)
kernel = ct.WinDLL("kernel32", use_last_error=True)
psapi = ct.WinDLL("psapi", use_last_error=True)
pdh = ct.WinDLL("pdh")


class Memory(ct.Structure):
    _fields_ = [("length", wt.DWORD), ("load", wt.DWORD)] + [
        (name, ct.c_ulonglong) for name in
        ("physical_total", "physical_available", "commit_limit", "commit_available", "virtual_total", "virtual_available", "extended")]


class ProcessMemory(ct.Structure):
    _fields_ = [("cb", wt.DWORD), ("page_faults", wt.DWORD)] + [
        (name, ct.c_size_t) for name in
        ("peak_working_set", "working_set", "peak_paged_pool", "paged_pool", "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private_bytes")]


class Io(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class PdhValue(ct.Structure):
    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class ProcessEntry(ct.Structure):
    _fields_ = [("size", wt.DWORD), ("usage", wt.DWORD), ("pid", wt.DWORD),
                ("heap", ct.c_size_t), ("module", wt.DWORD), ("threads", wt.DWORD),
                ("parent", wt.DWORD), ("priority", wt.LONG), ("flags", wt.DWORD),
                ("exe", wt.WCHAR * 260)]


kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
kernel.Process32FirstW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
kernel.Process32NextW.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel.OpenProcess.restype = wt.HANDLE
kernel.CloseHandle.argtypes = [wt.HANDLE]
kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]


def descendants(root_pid):
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    entries = {}
    entry = ProcessEntry()
    entry.size = ct.sizeof(entry)
    try:
        ok = kernel.Process32FirstW(snapshot, ct.byref(entry))
        while ok:
            entries[entry.pid] = (entry.parent, entry.exe)
            ok = kernel.Process32NextW(snapshot, ct.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in entries.items() if parent in selected}
        if expanded == selected:
            return {pid: entries.get(pid, (0, "unknown"))[1] for pid in selected}
        selected = expanded


kernel.GlobalMemoryStatusEx.argtypes = [ct.POINTER(Memory)]
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ct.POINTER(ProcessMemory), wt.DWORD]
kernel.GetProcessIoCounters.argtypes = [wt.HANDLE, ct.POINTER(Io)]
kernel.GetProcessTimes.argtypes = [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4
pdh.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
pdh.PdhAddEnglishCounterW.argtypes = [wt.HANDLE, wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
pdh.PdhCollectQueryData.argtypes = [wt.HANDLE]
pdh.PdhGetFormattedCounterValue.argtypes = [wt.HANDLE, wt.DWORD, ct.POINTER(wt.DWORD), ct.POINTER(PdhValue)]
pdh.PdhCloseQuery.argtypes = [wt.HANDLE]


def system_memory():
    data = Memory()
    data.length = ct.sizeof(data)
    if not kernel.GlobalMemoryStatusEx(ct.byref(data)):
        raise ct.WinError(ct.get_last_error())
    return {"available_ram_bytes": data.physical_available,
            "total_ram_bytes": data.physical_total,
            "system_commit_bytes": data.commit_limit - data.commit_available}


query = wt.HANDLE()
counters = {}
counter_errors = {}
if pdh.PdhOpenQueryW(None, 0, ct.byref(query)) == 0:
    for name, path in {
        "system_page_reads_per_sec": r"\Memory\Page Reads/sec",
        "system_pages_input_per_sec": r"\Memory\Pages Input/sec",
        "system_disk_read_bytes_per_sec": r"\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    }.items():
        handle = wt.HANDLE()
        status = pdh.PdhAddEnglishCounterW(query, path, 0, ct.byref(handle))
        if status == 0:
            counters[name] = handle
        else:
            counter_errors[name] = status
    pdh.PdhCollectQueryData(query)

model = json.loads((ROOT / "models/model.json").read_text(encoding="utf-8-sig"))
model_path = ROOT / "models" / model["filename"]
if model_path.stat().st_size != model["size_bytes"]:
    raise RuntimeError("Unexpected model size")
prompt = "In one sentence, name the capital of France."
(OUT / "prompt.txt").write_text(prompt, encoding="utf-8")
command = [str(ROOT / "build/llama-cpu/bin/Release/llama-cli.exe"),
           "--model", str(model_path), "--gpu-layers", "0", "--threads", "8",
           "--threads-batch", "8", "--ctx-size", "2048", "--batch-size", "128",
           "--ubatch-size", "128", "--predict", "128", "--seed", "42", "--temp", "0",
           "--load-mode", "mmap", "--cache-type-k", "f16", "--cache-type-v", "f16",
           "--fit", "off", "--single-turn", "--simple-io", "--perf", "--verbosity", "3",
           "--log-colors", "off", "--prompt", prompt]
if args.no_repack:
    command.append("--no-repack")
if args.no_warmup:
    command.append("--no-warmup")
manifest = {"command": command, "model": model, "prior_model_verification": json.loads(
    (ROOT / "models/download-status.json").read_text(encoding="utf-8-sig")),
    "time_limit_seconds": 300, "low_available_ram_bytes": 512 * 1024 * 1024,
    "warmup": "disabled" if args.no_warmup else "engine default enabled", "weight_repacking": not args.no_repack, "initial_memory": system_memory(),
    "paging_scope": "System-wide counters include unrelated programs; process faults include soft faults.",
    "counter_errors": counter_errors, "process_scope": "Runner and all observed descendants; working-set sum can double-count shared pages.",
    "cache_state": "Uncontrolled Windows file cache; not claimed to be cold."}
(OUT / "configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Results: {OUT}", flush=True)
start = time.perf_counter()
reason = "process_exit"
samples = []
low_count = 0
monitor_error = None
with (OUT / "stdout.txt").open("wb") as stdout, (OUT / "stderr.txt").open("wb") as stderr:
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    def stop_tree():
        owned = descendants(process.pid)
        for pid in sorted(owned, key=lambda p: p == process.pid):
            handle = kernel.OpenProcess(1, False, pid)
            if handle:
                try:
                    kernel.TerminateProcess(handle, 1)
                finally:
                    kernel.CloseHandle(handle)
        process.wait(timeout=30)
    try:
        with (OUT / "telemetry.jsonl").open("w", encoding="utf-8") as trace:
            while process.poll() is None:
                sample = {"elapsed_seconds": time.perf_counter() - start, **system_memory()}
                sample["processes"] = []
                for pid, name in descendants(process.pid).items():
                    handle = kernel.OpenProcess(0x410, False, pid)
                    if not handle:
                        continue
                    record = {"pid": pid, "name": name}
                    try:
                        pm = ProcessMemory()
                        pm.cb = ct.sizeof(pm)
                        if psapi.GetProcessMemoryInfo(handle, ct.byref(pm), pm.cb):
                            record.update({"working_set_bytes": pm.working_set, "peak_working_set_bytes": pm.peak_working_set,
                                           "private_bytes": pm.private_bytes, "page_fault_count": pm.page_faults})
                        io = Io()
                        if kernel.GetProcessIoCounters(handle, ct.byref(io)):
                            record.update({"process_read_bytes": io.read_bytes, "process_write_bytes": io.write_bytes})
                        times = [wt.FILETIME() for _ in range(4)]
                        if kernel.GetProcessTimes(handle, *(ct.byref(t) for t in times)):
                            record["cpu_seconds"] = sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 1e7
                    finally:
                        kernel.CloseHandle(handle)
                    sample["processes"].append(record)
                for name in ("working_set_bytes", "private_bytes", "page_fault_count", "process_read_bytes", "process_write_bytes", "cpu_seconds"):
                    sample[name] = sum(p.get(name, 0) for p in sample["processes"])
                if query and pdh.PdhCollectQueryData(query) == 0:
                    for name, counter in counters.items():
                        value = PdhValue()
                        if pdh.PdhGetFormattedCounterValue(counter, 0x200, None, ct.byref(value)) == 0 and value.status in (0, 1):
                            sample[name] = value.value
                samples.append(sample)
                trace.write(json.dumps(sample) + "\n")
                trace.flush()
                low_count = low_count + 1 if sample["available_ram_bytes"] < manifest["low_available_ram_bytes"] else 0
                if sample["elapsed_seconds"] >= 300 or low_count >= 3 or sample["available_ram_bytes"] < 128 * 1024 * 1024:
                    reason = "time_limit" if sample["elapsed_seconds"] >= 300 else "low_available_ram"
                    stop_tree()
                    break
                time.sleep(1)
        process.wait(timeout=30)
    except Exception as exc:
        monitor_error = repr(exc)
    finally:
        if process.poll() is None:
            try:
                stop_tree()
            except Exception as exc:
                monitor_error = f"{monitor_error or ''} Cleanup: {exc!r}"
        if query:
            pdh.PdhCloseQuery(query)
summary = {"stop_reason": reason, "exit_code": process.returncode, "monitor_error": monitor_error,
           "wall_seconds": time.perf_counter() - start, "sample_count": len(samples),
           "peak_sampled_tree_working_set_bytes": max((x.get("working_set_bytes", 0) for x in samples), default=0),
           "peak_private_bytes": max((x.get("private_bytes", 0) for x in samples), default=0),
           "minimum_available_ram_bytes": min((x["available_ram_bytes"] for x in samples), default=None),
           "last_sample": samples[-1] if samples else None,
           "counter_errors": counter_errors}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
