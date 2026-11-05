"""Run a Windows engine with a process-local 4 GiB resident cap by default."""
import ctypes as ct
from ctypes import wintypes as wt
import os
import subprocess
import sys


def main():
    limit = float(os.environ.get("SAPHRIEL_RESIDENT_GIB", "4"))
    if not 1 <= limit <= 8:
        raise ValueError("SAPHRIEL_RESIDENT_GIB must be between 1 and 8")
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ct.c_size_t, ct.c_size_t, wt.DWORD]
    process = subprocess.Popen(sys.argv[1:])
    try:
        handle = kernel.OpenProcess(0x100, False, process.pid)  # PROCESS_SET_QUOTA
        if not handle:
            raise ct.WinError(ct.get_last_error())
        try:
            if not kernel.SetProcessWorkingSetSizeEx(handle, 1024 * 1024, int(limit * 2**30), 0x6):
                raise ct.WinError(ct.get_last_error())
        finally:
            kernel.CloseHandle(handle)
        print(f"Resident cap: {limit:g} GiB; paging can make generation very slow.", flush=True)
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"Unable to launch safely: {exc}", file=sys.stderr)
        sys.exit(1)
