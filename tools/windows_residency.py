"""Read-only Windows working-set attribution; no target pages are read or locked."""
import bisect
import ctypes as ct
from ctypes import wintypes as wt
import time

kernel = ct.WinDLL("kernel32", use_last_error=True)
psapi = ct.WinDLL("psapi", use_last_error=True)


class Region(ct.Structure):
    _fields_ = [("base", ct.c_void_p), ("allocation", ct.c_void_p),
                ("allocation_protection", wt.DWORD), ("partition", wt.WORD),
                ("size", ct.c_size_t), ("state", wt.DWORD),
                ("protection", wt.DWORD), ("kind", wt.DWORD)]


kernel.VirtualQueryEx.argtypes = [wt.HANDLE, ct.c_void_p, ct.POINTER(Region), ct.c_size_t]
kernel.VirtualQueryEx.restype = ct.c_size_t
psapi.QueryWorkingSet.argtypes = [wt.HANDLE, ct.c_void_p, wt.DWORD]
psapi.GetMappedFileNameW.argtypes = [wt.HANDLE, ct.c_void_p, wt.LPWSTR, wt.DWORD]


def attribute_working_set(handle, model_name):
    start = time.perf_counter()
    regions = []
    names = {}
    address = 0
    region = Region()
    while kernel.VirtualQueryEx(handle, ct.c_void_p(address), ct.byref(region), ct.sizeof(region)):
        base = region.base or 0
        end = base + region.size
        if end <= address:
            break
        if region.state == 0x1000:
            category = {0x20000: "private", 0x1000000: "image", 0x40000: "other_mapped"}.get(region.kind, "unknown")
            if region.kind == 0x40000:
                if region.allocation not in names:
                    buffer = ct.create_unicode_buffer(32768)
                    length = psapi.GetMappedFileNameW(handle, ct.c_void_p(region.base), buffer, len(buffer))
                    names[region.allocation] = buffer.value[:length] if length else ""
                if names[region.allocation].lower().endswith("\\" + model_name.lower()):
                    category = "model_file"
            regions.append((base, end, category))
        address = end
    starts = [r[0] for r in regions]
    for size in (1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024):
        buffer = ct.create_string_buffer(size)
        if psapi.QueryWorkingSet(handle, buffer, size):
            break
        error = ct.get_last_error()
        if error != 24:
            return {"error": error, "measurement_seconds": time.perf_counter() - start}
    else:
        return {"error": "working_set_buffer_limit", "measurement_seconds": time.perf_counter() - start}
    count = ct.c_size_t.from_buffer(buffer).value
    if (count + 1) * ct.sizeof(ct.c_size_t) > size:
        return {"error": "invalid_working_set_count"}
    pages = (ct.c_size_t * count).from_buffer(buffer, ct.sizeof(ct.c_size_t))
    totals = dict.fromkeys(("model_file", "private", "image", "other_mapped", "unknown"), 0)
    for flags in pages:
        page = flags & ~0xFFF
        index = bisect.bisect_right(starts, page) - 1
        category = regions[index][2] if index >= 0 and page < regions[index][1] else "unknown"
        totals[category] += 4096
    return {"resident_bytes": totals, "total_enumerated_resident_bytes": count * 4096,
            "measurement_seconds": time.perf_counter() - start,
            "note": "Region map and working set sampled sequentially; not an atomic snapshot."}
