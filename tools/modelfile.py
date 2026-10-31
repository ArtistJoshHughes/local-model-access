#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Josh Hughes
"""modelfile.py - Local Model Access (LMA): read the header of a GGUF or safetensors model file (never the tensor data) and work out
its identity (SPEC.md section 3).

  python modelfile.py info FILE                     format, name, architecture, metadata and every tensor (dtype, shape, offset, size)
  python modelfile.py identity FILE [FILE ...]      {"file", "size", "sha256"} per file: streams each file once (the sha256 Hugging Face lists)
  python modelfile.py hf REPO FILE [FILE ...] [--revision REV]
                                                    the approved-models entry Hugging Face's metadata gives: {"repo", "revision" (the commit
                                                    sha), "file", "sha256", "size"} for one file, or "files": [...] for a model in several
                                                    files (split GGUF parts, safetensors shards). Metadata only: nothing is downloaded.
                                                    HF_TOKEN is used if set (gated repositories).

Standard library only (Python 3.8+). The same readers exist in JavaScript in lma/formats.js."""
import hashlib, json, os, struct, sys, urllib.parse, urllib.request

MiB = 1 << 20
# ggml tensor types: id -> (dtype, elements per block, bytes per block)
GGML_TYPES = {0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20), 6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24),
              8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40), 10: ("q2_k", 256, 84), 11: ("q3_k", 256, 110), 12: ("q4_k", 256, 144),
              13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210), 15: ("q8_k", 256, 292), 16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74),
              18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110), 22: ("iq2_s", 256, 82),
              23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
              29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66), 39: ("mxfp4", 32, 17),
              40: ("nvfp4", 64, 36), 41: ("q1_0", 128, 18)}
ST_DTYPES = {"F64": ("f64", 8), "F32": ("f32", 4), "F16": ("f16", 2), "BF16": ("bf16", 2), "I64": ("i64", 8), "U64": ("u64", 8),
             "I32": ("i32", 4), "U32": ("u32", 4), "I16": ("i16", 2), "U16": ("u16", 2), "I8": ("i8", 1), "U8": ("u8", 1),
             "BOOL": ("bool", 1), "F8_E4M3": ("f8_e4m3", 1), "F8_E5M2": ("f8_e5m2", 1), "F8_E8M0": ("f8_e8m0", 1)}
# GGUF metadata value types: id -> (name, struct format)
GV = {0: ("u8", "<B"), 1: ("i8", "<b"), 2: ("u16", "<H"), 3: ("i16", "<h"), 4: ("u32", "<I"), 5: ("i32", "<i"), 6: ("f32", "<f"),
      7: ("bool", "<?"), 10: ("u64", "<Q"), 11: ("i64", "<q"), 12: ("f64", "<d")}


class FormatError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code = code


def _prod(xs):
    n = 1
    for x in xs: n *= x
    return n


class _R:
    """forward reader over a file object"""
    def __init__(self, f, size): self.f, self.size = f, size

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n: raise FormatError("truncated", "the file ends inside its header")
        return b

    def unpack(self, fmt): return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def str(self, limit=64 * MiB):
        n = self.unpack("<Q")
        if n > limit: raise FormatError("bad-header", "a string in the header is implausibly long")
        return self.take(n).decode("utf-8", errors="replace")

    def skip(self, n): self.f.seek(n, 1)


def _gguf_value(r, t, max_array=4096, max_strings=64, depth=0):
    if t == 8: return r.str()
    if t in GV: return r.unpack(GV[t][1])
    if t != 9 or depth > 2: raise FormatError("bad-header", "unknown GGUF metadata type %d" % t)
    et, n = r.unpack("<I"), r.unpack("<Q")
    if et == 8:
        if n > max_strings:
            for _ in range(n): r.skip(r.unpack("<Q"))
            return {"skipped": True, "of": "string", "length": n}
        return [r.str() for _ in range(n)]
    if et in GV:
        sz = struct.calcsize(GV[et][1])
        if n > max_array: r.skip(n * sz); return {"skipped": True, "of": GV[et][0], "length": n}
        return list(struct.unpack("<%d%s" % (n, GV[et][1][1]), r.take(n * sz)))
    return [_gguf_value(r, et, max_array, max_strings, depth + 1) for _ in range(n)]


def read_gguf(f, size):
    r = _R(f, size)
    if r.take(4) != b"GGUF": raise FormatError("not-gguf", "not a GGUF file")
    version = r.unpack("<I")
    if version not in (2, 3): raise FormatError("version", "GGUF version %d is not supported (2 and 3 are)" % version)
    n_t, n_kv = r.unpack("<Q"), r.unpack("<Q")
    if n_t > 1e6 or n_kv > 1e6: raise FormatError("bad-header", "implausible GGUF counts")
    meta = {}
    for _ in range(n_kv):
        key = r.str(1 << 16); meta[key] = _gguf_value(r, r.unpack("<I"))
    raw = []
    for _ in range(n_t):
        name = r.str(1 << 16); nd = r.unpack("<I")
        if not 1 <= nd <= 8: raise FormatError("bad-header", "tensor %s has %d dimensions" % (name, nd))
        ne = [r.unpack("<Q") for _ in range(nd)]
        raw.append((name, ne, r.unpack("<I"), r.unpack("<Q")))
    align = meta.get("general.alignment", 32)
    if not isinstance(align, int) or align < 1 or align % 8: raise FormatError("bad-header", "general.alignment must be a multiple of 8")
    header_end = f.tell(); data_start = -(-header_end // align) * align
    tensors = []
    for name, ne, t, rel in raw:
        if rel % align: raise FormatError("bad-header", "tensor %s is not aligned" % name)
        info = GGML_TYPES.get(t); tsize = None
        if info:
            if ne[0] % info[1]: raise FormatError("bad-header", "tensor %s: a row is not a whole number of %s blocks" % (name, info[0]))
            tsize = _prod(ne) // info[1] * info[2]
        tensors.append({"name": name, "dtype": info[0] if info else "ggml_%d" % t, "ggmlType": t, "shape": ne[::-1], "offset": data_start + rel, "size": tsize})
    by_off = sorted(tensors, key=lambda x: x["offset"])
    for i, x in enumerate(by_off):
        if x["size"] is None:
            x["size"] = (by_off[i + 1]["offset"] if i + 1 < len(by_off) else size) - x["offset"]; x["sizeFromGap"] = True
    for x in tensors:
        if x["offset"] + x["size"] > size: raise FormatError("truncated", "tensor %s runs past the end of the file" % x["name"])
    return {"format": "gguf", "version": version, "metadata": meta, "tensors": tensors, "dataStart": data_start, "alignment": align,
            "headerBytes": header_end, "name": meta.get("general.name"), "architecture": meta.get("general.architecture")}


def read_safetensors(f, size):
    n = struct.unpack("<Q", f.read(8))[0]
    if n < 2 or n > 100 * MiB or 8 + n > size: raise FormatError("not-safetensors", "not a safetensors file")
    try: j = json.loads(f.read(n).decode("utf-8"))
    except Exception: raise FormatError("bad-header", "the safetensors header is not valid JSON")
    if not isinstance(j, dict): raise FormatError("bad-header", "the safetensors header is not a JSON object")
    data_start = 8 + n; data_size = size - data_start; meta = j.get("__metadata__") or {}; tensors = []
    for name, t in j.items():
        if name == "__metadata__": continue
        b, e = t["data_offsets"]
        if not (0 <= b <= e): raise FormatError("bad-header", "tensor %s has bad offsets" % name)
        if e > data_size: raise FormatError("truncated", "tensor %s runs past the end of the file" % name)
        info = ST_DTYPES.get(t["dtype"])
        if info and _prod(t["shape"]) * info[1] != e - b: raise FormatError("bad-header", "tensor %s: its size does not match its shape" % name)
        tensors.append({"name": name, "dtype": info[0] if info else str(t["dtype"]).lower(), "shape": list(t["shape"]), "offset": data_start + b, "size": e - b})
    return {"format": "safetensors", "version": None, "metadata": meta, "tensors": tensors, "dataStart": data_start, "alignment": 1,
            "headerBytes": data_start, "name": meta.get("general.name") or meta.get("name"), "architecture": meta.get("general.architecture") or meta.get("architecture")}


def sniff(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f: u = f.read(16)
    if size < 9: return None
    if u[:4] == b"GGUF": return "gguf"
    n = struct.unpack("<Q", u[:8])[0]
    if 2 <= n <= 100 * MiB and 8 + n <= size and u[8:9] == b"{": return "safetensors"
    return None


def read_header(path):
    fmt = sniff(path); size = os.path.getsize(path)
    if not fmt: raise FormatError("not-model", "this is not a GGUF or safetensors model file")
    with open(path, "rb") as f:
        m = read_gguf(f, size) if fmt == "gguf" else read_safetensors(f, size)
    m["fileSize"] = size
    return m


def identity(path, piece=8 * MiB):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(piece)
            if not b: break
            h.update(b)
    return {"file": os.path.basename(path), "size": os.path.getsize(path), "sha256": h.hexdigest()}


def hf_identity(repo, files, revision="main"):
    """the approved-models entry Hugging Face's metadata gives for one or more files of a model repo: the commit sha of `revision` and
    each file's LFS sha256 + size (metadata only)"""
    files = [files] if isinstance(files, str) else list(files)
    url = "https://huggingface.co/api/models/%s/revision/%s?blobs=true" % (repo, urllib.parse.quote(revision, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "lma-modelfile.py"})
    # the token goes to huggingface.co only: an unredirected header is not copied onto a redirect to another host
    if os.environ.get("HF_TOKEN"): req.add_unredirected_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(req, timeout=30) as r: j = json.load(r)
    out = []
    for file in files:
        sib = next((s for s in j.get("siblings", []) if s.get("rfilename") == file), None)
        if not sib: raise SystemExit("%s has no file %s at %s" % (repo, file, revision))
        lfs = sib.get("lfs") or {}
        if not lfs.get("sha256"): raise SystemExit("%s is not stored with LFS, so Hugging Face lists no sha256 for it: download it and run 'identity'" % file)
        out.append({"file": file, "sha256": lfs["sha256"], "size": lfs.get("size", sib.get("size"))})
    fmt = "gguf" if all(f["file"].endswith(".gguf") for f in out) else "safetensors" if all(f["file"].endswith(".safetensors") for f in out) else None
    e = {"repo": repo, "revision": j["sha"]}
    if fmt: e["format"] = fmt
    if len(out) == 1: e.update(out[0])
    else: e["files"] = out
    return e


def safe(s):
    """text from a file, printable: control characters (terminal escape sequences) are shown escaped instead of acting on the terminal"""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in str(s))


def main(argv):
    if len(argv) < 2 or argv[0] not in ("info", "identity", "hf"): raise SystemExit(__doc__)
    if argv[0] == "info":
        m = read_header(argv[1])
        print(json.dumps({k: v for k, v in m.items() if k != "tensors"}, indent=1, default=str))
        for t in m["tensors"]: print("%-48s %-8s %-22s offset %-12d size %d" % (safe(t["name"]), safe(t["dtype"]), t["shape"], t["offset"], t["size"]))
    elif argv[0] == "identity":
        ids = [identity(p) for p in argv[1:]]
        print(json.dumps(ids[0] if len(ids) == 1 else ids, indent=1))
    else:
        rev = "main"
        if "--revision" in argv: i = argv.index("--revision"); rev = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
        if len(argv) < 3: raise SystemExit(__doc__)
        print(json.dumps(hf_identity(argv[1], argv[2:], rev), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
