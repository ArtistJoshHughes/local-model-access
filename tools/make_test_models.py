#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Josh Hughes
"""make_test_models.py - Local Model Access (LMA) test models: real GGUF v3 and safetensors files (single, split and sharded) holding
random int8 weight matrices, with the exact answers of a matrix-vector product stored in their metadata, so a reader can prove it found
every tensor and a GPU loader can prove it read every byte. The weights come from SHAKE-256: every run writes byte-identical files.

  python make_test_models.py vectors OUTDIR [--skip-big]       the conformance test vectors + OUTDIR/vectors.json (see CONFORMANCE.md);
                                                               the 1 GB vector's identity is computed in memory, not written
  python make_test_models.py OUTDIR [--big] [--mid] [--ext EXT]  test models for the demo + OUTDIR/models.json (--ext .bin names the
                                                               served copies, for a server that sends .gguf as text/plain)
  python make_test_models.py identity KEY                      one test model's identity, computed in memory (nothing written)
  python make_test_models.py approve OUTDIR CONFIG [...]       add OUTDIR/models.json's models to site configs' approved lists

  small    lma-test-1m.gguf                        GGUF v3, 3 layers of 512 x 512                (about 0.8 MB)
  st       lma-test-st.safetensors                 safetensors, 2 layers of 256 x 512           (about 0.27 MB)
  split    lma-test-split-0000N-of-00002.gguf      a split GGUF in 2 parts, 3 layers of 512 x 512
  sharded  lma-test-sharded-0000N-of-00002.safetensors + lma-test-sharded.safetensors.index.json
  pack     lma-test-pack/  a pocketflux.pack/1 (extension format) with every encoding: manifest.json + 2 shards + 1 JSON file
  mid      lma-test-64m.gguf                       GGUF v3, 4 layers of 4096 x 4096             (64 MB; --mid)
  big      lma-test-1g.gguf                        GGUF v3, 63 layers of 4096 x 4096            (about 1.06 GB; --big)

Tensors: blk.N.weight i8 [rows, cols] (GGUF dims [cols, rows]), blk.N.scale f32 [rows], input.x i32 [cols].
Answers (metadata lma-test-gemv.check.*): acc[r] = sum_c weight[r, c] * x[c] exactly (int32); y[r] = acc[r] * scale[r];
sum[N] = sum_r acc[r]; wsum[N] = sum_r acc[r] * (r % 7 + 1). Needs numpy."""
import hashlib, json, os, struct, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelfile  # noqa: E402

PRESETS = {
    "small": dict(base="lma-test-1m", fmt="gguf", parts=1, layers=3, rows=512, cols=512, label="small GGUF (under 1 MB)"),
    "st": dict(base="lma-test-st", fmt="safetensors", parts=1, layers=2, rows=256, cols=512, label="small safetensors (under 1 MB)"),
    "split": dict(base="lma-test-split", fmt="gguf", parts=2, layers=3, rows=512, cols=512, label="split GGUF (2 parts)"),
    "sharded": dict(base="lma-test-sharded", fmt="safetensors", parts=2, layers=2, rows=256, cols=512, label="sharded safetensors (2 shards + index)"),
    "mid": dict(base="lma-test-64m", fmt="gguf", parts=1, layers=4, rows=4096, cols=4096, label="mid GGUF (64 MB)"),
    "big": dict(base="lma-test-1g", fmt="gguf", parts=1, layers=63, rows=4096, cols=4096, label="big GGUF (about 1 GB)"),
}
ARCH = "lma-test-gemv"
RULE = "acc[r] = sum_c weight[r,c] * x[c] (exact int32); y[r] = acc[r] * scale[r]; sum = sum_r acc[r]; wsum = sum_r acc[r] * (r % 7 + 1)"
IDENT = {"repo": "local/lma-test-vectors", "revision": "make_test_models.py@2"}


def _shake(tag, n): return hashlib.shake_256(("lma-test-vectors/" + tag).encode()).digest(n)


def weight(p, l):
    return np.frombuffer(_shake("%s/blk.%d.weight" % (p["base"], l), p["rows"] * p["cols"]), dtype=np.int8).reshape(p["rows"], p["cols"])


def scale(p, l):
    u = np.frombuffer(_shake("%s/blk.%d.scale" % (p["base"], l), p["rows"] * 2), dtype="<u2").astype(np.float64)
    return ((u / 65535.0 * 0.5 + 0.75) / 127.0).astype("<f4")


def xvec(cols): return ((np.arange(cols, dtype=np.int64) * 37 + 11) % 5 - 2).astype("<i4")


def checks(p):
    x = xvec(p["cols"]).astype(np.int64); wr = (np.arange(p["rows"]) % 7 + 1).astype(np.int64); s, ws = [], []
    for l in range(p["layers"]):
        acc = weight(p, l).astype(np.int64) @ x
        assert np.abs(acc).max() < 2 ** 31
        s.append(int(acc.sum())); ws.append(int((acc * wr).sum()))
    return s, ws


def tensors(p):
    """(name, ggml type, gguf dims, nbytes, bytes producer) in file order"""
    out = []
    for l in range(p["layers"]):
        out.append(("blk.%d.weight" % l, 24, [p["cols"], p["rows"]], p["rows"] * p["cols"], lambda l=l: weight(p, l).tobytes()))
        out.append(("blk.%d.scale" % l, 0, [p["rows"]], p["rows"] * 4, lambda l=l: scale(p, l).tobytes()))
    out.append(("input.x", 26, [p["cols"]], p["cols"] * 4, lambda: xvec(p["cols"]).tobytes()))
    return out


class _Out:
    """a file being written (or, with path None, only hashed and counted), hashed on the way out so a big file is never read back"""
    def __init__(self, path):
        self.path, self.h, self.n = path, hashlib.sha256(), 0
        self.f = open(str(path) + ".part", "wb") if path else None

    def write(self, b):
        self.h.update(b); self.n += len(b)
        if self.f: self.f.write(b)

    def close(self):
        if self.f:
            self.f.flush(); os.fsync(self.f.fileno()); self.f.close(); os.replace(str(self.path) + ".part", self.path)
        return {"size": self.n, "sha256": self.h.hexdigest()}


# ---------------- GGUF v3 writer (just what the test files need) ----------------
def _gs(s): b = s.encode("utf-8"); return struct.pack("<Q", len(b)) + b


def _kv(key, t, v):
    out = _gs(key) + struct.pack("<I", t)
    if t == 8: return out + _gs(v)
    if t in (2, 4, 5): return out + struct.pack({2: "<H", 4: "<I", 5: "<i"}[t], v)
    if t == 9:   # (element type, values)
        et, vals = v; out += struct.pack("<IQ", et, len(vals))
        return out + b"".join(_gs(x) for x in vals) if et == 8 else out + struct.pack("<%d%s" % (len(vals), {11: "q", 4: "I"}[et]), *vals)
    raise ValueError(t)


def write_gguf(path, meta, tens, align=32):
    out = _Out(path)
    hdr = struct.pack("<4sIQQ", b"GGUF", 3, len(tens), len(meta)) + b"".join(_kv(*m) for m in meta)
    off = 0; infos = []
    for name, t, dims, n, _ in tens:
        infos.append(_gs(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) + struct.pack("<IQ", t, off))
        off += -(-n // align) * align
    hdr += b"".join(infos); hdr += b"\0" * (-len(hdr) % align)
    out.write(hdr)
    for name, t, dims, n, make in tens:
        b = make(); assert len(b) == n, name
        out.write(b); out.write(b"\0" * (-n % align))
    return out.close()


def write_safetensors(path, meta, tens):
    rev = {24: "I8", 0: "F32", 26: "I32"}; j = {"__metadata__": meta} if meta else {}; off = 0
    for name, t, dims, n, _ in tens:
        j[name] = {"dtype": rev[t], "shape": dims[::-1], "data_offsets": [off, off + n]}; off += n
    h = json.dumps(j, separators=(",", ":")).encode("utf-8"); h += b" " * (-len(h) % 8)
    out = _Out(path); out.write(struct.pack("<Q", len(h))); out.write(h)
    for *_, n, make in tens:
        b = make(); assert len(b) == n
        out.write(b)
    return out.close()


def _names(p, ext):
    real = ".gguf" if p["fmt"] == "gguf" else ".safetensors"
    if p["parts"] == 1: return [(p["base"] + (ext or real), p["base"] + real)]
    return [("%s-%05d-of-%05d%s" % (p["base"], i + 1, p["parts"], ext or real), "%s-%05d-of-%05d%s" % (p["base"], i + 1, p["parts"], real)) for i in range(p["parts"])]


def make(outdir, key, ext=None, dry=False):
    """write one test model (dry: hash it in memory, write nothing). -> its record, with an identity per file"""
    p = PRESETS[key]; s, ws = checks(p); tens = tensors(p); names = _names(p, ext)
    name, desc = "LMA test vector (%s)" % key, "Random int8 weights for testing Local Model Access readers and GPU loaders. Not a language model."
    k = -(-len(tens) // p["parts"]); groups = [tens[i * k:(i + 1) * k] for i in range(p["parts"])]
    files = []
    for i, (served, real) in enumerate(names):
        path = None if dry else Path(outdir) / served
        if p["fmt"] == "gguf":
            meta = [("general.architecture", 8, ARCH), ("general.name", 8, name), ("general.description", 8, desc), ("general.license", 8, "CC0-1.0"),
                    (ARCH + ".block_count", 4, p["layers"]), (ARCH + ".rows", 4, p["rows"]), (ARCH + ".cols", 4, p["cols"]),
                    (ARCH + ".check.rule", 8, RULE), (ARCH + ".check.sum", 9, (11, s)), (ARCH + ".check.wsum", 9, (11, ws))] if i == 0 else []
            if p["parts"] > 1:   # the keys llama.cpp's gguf-split writes into every part
                meta += [("split.no", 2, i), ("split.count", 2, p["parts"]), ("split.tensors.count", 5, len(tens))]
            ident = write_gguf(path, meta, groups[i])
        else:
            meta = {"general.architecture": ARCH, "general.name": name, "general.description": desc, "general.license": "CC0-1.0",
                    ARCH + ".block_count": str(p["layers"]), ARCH + ".rows": str(p["rows"]), ARCH + ".cols": str(p["cols"]),
                    ARCH + ".check.rule": RULE, ARCH + ".check.sum": json.dumps(s), ARCH + ".check.wsum": json.dumps(ws)} if i == 0 else {}
            ident = write_safetensors(path, meta, groups[i])
        files.append({"file": served, "saveAs": real, **ident})
    rec = {"key": key, "format": p["fmt"], "label": p["label"], "name": name, "architecture": ARCH, "layers": p["layers"], "rows": p["rows"], "cols": p["cols"],
           "tensors": len(tens), "check": {"rule": RULE, "sum": s, "wsum": ws}, "files": files}
    if p["fmt"] == "safetensors" and p["parts"] > 1:
        idx = {"metadata": {"total_size": sum(t[3] for t in tens)}, "weight_map": {t[0]: names[i][1] for i, g in enumerate(groups) for t in g}}
        b = (json.dumps(idx, indent=2) + "\n").encode("utf-8"); iname = p["base"] + ".safetensors.index.json"
        if not dry: (Path(outdir) / iname).write_bytes(b)
        rec["index"] = {"file": iname, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if len(files) == 1: rec.update({"file": files[0]["file"], "saveAs": files[0]["saveAs"], "size": files[0]["size"], "sha256": files[0]["sha256"]})
    if not dry:   # our own reader must accept what we wrote
        got = sum(len(modelfile.read_header(str(Path(outdir) / f["file"]))["tensors"]) for f in files)
        assert got == len(tens), (key, got)
    return rec


def identity(rec):
    """the approved-models entry for a test model (the same shape as a Hugging Face identity)"""
    e = dict(IDENT, name=rec["name"], format=rec["format"])
    if len(rec["files"]) == 1: e.update(file=rec["files"][0]["saveAs"], sha256=rec["files"][0]["sha256"], size=rec["files"][0]["size"])
    else: e["files"] = [{"file": f["saveAs"], "sha256": f["sha256"], "size": f["size"]} for f in rec["files"]]
    e.update(capabilities=["test"], license="CC0-1.0", note="test vector: random int8 weights (tools/make_test_models.py)")
    return e


def approve(config, entries):
    """add (or update) test models in a site config's approved list, matched by name"""
    p = Path(config); c = json.loads(p.read_text(encoding="utf-8")); models = c.setdefault("models", [])
    for rec in entries:
        ident = identity(rec); i = next((k for k, m in enumerate(models) if m.get("name") == ident["name"]), None)
        if i is None: models.append(ident)
        else: models[i] = ident
    p.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------- pocketflux.pack/1 (the extension format, docs/QUANT-q4g32a.md): a tiny pack with every encoding ----------------
def q4g32a(v):
    """v float32 [N, K] -> codes [N, K/32, 32] (0..15), scale and min [N, K/32] as f16, as p0pc0rn-kernels FORMAT.md defines it"""
    N, K = v.shape; x = v.reshape(N, K // 32, 32)
    mn = x.min(-1).astype(np.float16); mnf = mn.astype(np.float32)
    s = ((x.max(-1) - mnf) / np.float32(15)).astype(np.float16); s = np.where(s == 0, np.float16(1), s)
    q = np.clip(np.round((x - mnf[..., None]) / s.astype(np.float32)[..., None]), 0, 15).astype(np.uint8)
    return q, s, mn


def q8g64s(v):
    N, K = v.shape; x = v.reshape(N, K // 64, 64)
    s = (np.abs(x).max(-1) / np.float32(127)).astype(np.float16); s = np.where(s == 0, np.float16(1), s)
    q = np.clip(np.round(x / s.astype(np.float32)[..., None]), -128, 127).astype(np.int8)
    return q, s


def deq4(q, s, mn): return (q.astype(np.float32) * s.astype(np.float32)[..., None] + mn.astype(np.float32)[..., None]).reshape(q.shape[0], -1)
def deq8(q, s): return (q.astype(np.float32) * s.astype(np.float32)[..., None]).reshape(q.shape[0], -1)


def enc_q4_t64(q, s, mn):
    out = bytearray()
    for t in range(q.shape[0] // 64):
        for g in range(q.shape[1]):
            r = slice(64 * t, 64 * t + 64)
            out += (q[r, g, 0::2] | (q[r, g, 1::2] << 4)).astype(np.uint8).tobytes()
            out += (s[r, g].view(np.uint16).astype(np.uint32) | (mn[r, g].view(np.uint16).astype(np.uint32) << 16)).astype("<u4").tobytes()
    return bytes(out)


def enc_q8_t64(q, s):
    out = bytearray()
    for t in range(q.shape[0] // 64):
        for g in range(q.shape[1]):
            r = slice(64 * t, 64 * t + 64)
            out += q[r, g, :].astype(np.int8).tobytes() + s[r, g].astype("<f2").tobytes()
    return bytes(out)


def enc_q4_rows(q, s, mn):
    return b"".join((q[n, :, 0::2] | (q[n, :, 1::2] << 4)).astype(np.uint8).tobytes() + np.stack([s[n].view(np.uint16), mn[n].view(np.uint16)], -1).astype("<u2").tobytes() for n in range(q.shape[0]))


def enc_q8_rows(q, s):
    return b"".join(q[n].astype(np.int8).tobytes() + s[n].astype("<f2").tobytes() for n in range(q.shape[0]))


def _vals(tag, n, lo=-0.5, hi=0.5):
    u = np.frombuffer(_shake("pack/" + tag, n * 2), dtype="<u2").astype(np.float64)
    return (lo + (hi - lo) * u / 65535.0).astype(np.float32)


def make_pack(outdir):
    """write test-vectors/lma-test-pack/ (manifest.json + 2 shards + 1 JSON file) and return its vectors.json record"""
    root = Path(outdir) / "lma-test-pack"; (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    W1 = _vals("gate_up", 128 * 64).reshape(128, 64); W1[0, 0:32] = 0.25          # one constant group: scale 0 -> 1
    W2 = _vals("qkv", 64 * 128).reshape(64, 128)
    R1 = _vals("ple", 16 * 64).reshape(16, 64); R2 = _vals("rows8", 8 * 128).reshape(8, 128)
    N1 = _vals("norm", 64, 0.5, 1.5); H1 = _vals("pos", 8 * 16).astype(np.float16)
    U1 = np.frombuffer(_shake("pack/order", 64 * 4), dtype="<u4") % 64
    q, s, mn = q4g32a(W1); t_w1 = (enc_q4_t64(q, s, mn), deq4(q, s, mn), "q4g32a.t64", [128, 64])
    q8, s8 = q8g64s(W2); t_w2 = (enc_q8_t64(q8, s8), deq8(q8, s8), "q8g64s.t64", [64, 128])
    q, s, mn = q4g32a(R1); t_r1 = (enc_q4_rows(q, s, mn), deq4(q, s, mn), "q4g32a.rows", [16, 64])
    q8, s8 = q8g64s(R2); t_r2 = (enc_q8_rows(q8, s8), deq8(q8, s8), "q8g64s.rows", [8, 128])
    t_n1 = (N1.astype("<f4").tobytes(), N1, "f32", [64]); t_h1 = (H1.astype("<f2").tobytes(), H1.astype(np.float32), "f16", [8, 16])
    t_u1 = (U1.astype("<u4").tobytes(), U1, "u32", [64])
    layout = {"lm.global": [("layers.0.mlp.gate_up_proj.weight", t_w1), ("layers.0.self_attn.qkv_proj.weight", t_w2), ("norm.weight", t_n1),
                            ("position_table", t_h1), ("token_ordering", t_u1)],
              "ple.rows": [("embed_tokens_per_layer.weight", t_r1), ("embed_rows_q8", t_r2)]}
    files = {"lm.global": "lm.global.pfw", "ple.rows": "ple.rows.pfw"}
    tensors, expected, shards, CB = {}, {}, [], 4096
    for sh, items in layout.items():
        data = bytearray()
        for name, (b, deq, enc, shape) in items:
            data += b"\0" * (-len(data) % 256)   # every tensor starts at a 256-byte boundary, as in the published packs
            tensors[name] = {"shard": sh, "offset": len(data), "bytes": len(b), "shape": shape, "enc": enc, "src": [{"name": name}]}
            flat = np.asarray(deq).reshape(-1)
            expected[name] = {"enc": enc, "shape": shape, "first": [float(x) for x in flat[:8]], "sum": float(np.sum(flat.astype(np.float64)))}
            data += b
        (root / files[sh]).write_bytes(bytes(data))
        shards.append({"name": sh, "file": files[sh], "kind": "lm_global" if sh == "lm.global" else "ple_rows", "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(), "chunk_bytes": CB,
                       "chunk_sha256": [hashlib.sha256(data[i:i + CB]).hexdigest() for i in range(0, len(data), CB)]})
    meta = (json.dumps({"note": "a pack may carry other files (a tokenizer, a chat template); the manifest lists and hashes them"}, indent=1) + "\n").encode()
    (root / "tokenizer" / "tokenizer_meta.json").write_bytes(meta)
    shards.append({"name": "tokenizer.meta", "file": "tokenizer/tokenizer_meta.json", "kind": "file", "bytes": len(meta), "sha256": hashlib.sha256(meta).hexdigest(),
                   "chunk_bytes": CB, "chunk_sha256": [hashlib.sha256(meta).hexdigest()]})
    man = {"format": "pocketflux.pack", "format_version": 1, "pack_id": "lma-test-pack", "pack_version": "1",
           "model": {"id": "lma-test-pack", "arch": "lma-test"},
           "encodings": {"q4g32a.t64": "docs/QUANT-q4g32a.md", "q8g64s.t64": "docs/QUANT-q4g32a.md", "q4g32a.rows": "docs/QUANT-q4g32a.md",
                         "q8g64s.rows": "docs/QUANT-q4g32a.md", "f16": "docs/QUANT-q4g32a.md", "f32": "docs/QUANT-q4g32a.md", "u32": "little-endian uint32"},
           "quant_recipe": {"name": "lma-test", "rules": [["qkv_proj|rows8", "q8g64s"], [".", "q4g32a"]]},
           "sources": [], "license": {"spdx": "CC0-1.0"}, "shards": shards, "tensors": tensors}
    mb = (json.dumps(man, indent=1) + "\n").encode(); (root / "manifest.json").write_bytes(mb)
    pf = [{"file": "lma-test-pack/manifest.json", "size": len(mb), "sha256": hashlib.sha256(mb).hexdigest()}] + \
         [{"file": "lma-test-pack/" + s["file"], "size": s["bytes"], "sha256": s["sha256"]} for s in shards]
    ident = dict(IDENT, name="LMA test vector (pocketflux.pack)", format="pocketflux.pack/1", manifest={"file": "manifest.json", "sha256": pf[0]["sha256"], "size": len(mb)},
                 size=sum(s["bytes"] for s in shards), shards=len(shards), quant="lma-test (q4g32a + q8g64s)", capabilities=["test"], license="CC0-1.0",
                 note="test vector: random weights in every pocketflux.pack/1 encoding (tools/make_test_models.py)")
    return {"id": "pack", "label": "pocketflux.pack/1 (extension, SPEC.md 2.2)", "format": "pocketflux.pack", "files": pf, "identity": ident,
            "chunks": {s["file"]: len(s["chunk_sha256"]) for s in shards}, "tensors": expected}


def header_facts(outdir, rec):
    """what a conforming reader must find: name, architecture, tensor count, and the first and last tensor of every part"""
    facts = []
    for i, f in enumerate(rec["files"]):
        m = modelfile.read_header(str(Path(outdir) / f["file"]))
        for t in (m["tensors"][0], m["tensors"][-1]):
            facts.append({"part": i, "name": t["name"], "dtype": t["dtype"], "shape": t["shape"], "offset": t["offset"], "size": t["size"]})
    return facts


def vectors(outdir, skip_big=False):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True); vs = []
    for key in ("small", "st", "split", "sharded"):
        rec = make(out, key); print("%-8s %s" % (key, ", ".join("%s %d B" % (f["file"], f["size"]) for f in rec["files"])))
        vs.append({"id": key, "label": rec["label"], "format": rec["format"], "files": [{k: f[k] for k in ("file", "size", "sha256")} for f in rec["files"]],
                   **({"index": rec["index"]} if "index" in rec else {}),
                   "header": {"name": rec["name"], "architecture": rec["architecture"], "tensors": rec["tensors"], "parts": len(rec["files"]), "tensorsSeen": header_facts(out, rec)},
                   "gemv": dict(rec["check"], layers=rec["layers"], rows=rec["rows"], cols=rec["cols"], x="input.x", weight="blk.{N}.weight", scale="blk.{N}.scale"),
                   "identity": identity(rec)})
    doc = {"about": "Local Model Access (LMA) conformance test vectors. See CONFORMANCE.md. Each vector is one model (one or more files) with its "
                    "expected identity and header facts; 'refuse' lists inputs a conforming reader must reject, with the error code.",
           "spec": "1.0", "vectors": vs,
           "refuse": [{"id": "not-a-model", "input": "4096 random bytes", "code": "not-model"},
                      {"id": "truncated", "input": "lma-test-1m.gguf without its last 1000 bytes", "code": "truncated"},
                      {"id": "header-cut", "input": "the first 200 bytes of lma-test-1m.gguf", "code": "truncated"},
                      {"id": "bad-version", "input": "lma-test-1m.gguf with bytes 4-7 set to 09 00 00 00", "code": "version"},
                      {"id": "missing-part", "input": "only part 1 of the split GGUF", "code": "missing-part"},
                      {"id": "missing-shard", "input": "shard 1 of the sharded safetensors with its index", "code": "missing-part"},
                      {"id": "changed-byte", "input": "lma-test-1m.gguf with byte 400000 changed", "result": "the header reads, the sha256 matches no approved identity: not approved"},
                      {"id": "pack-missing-shard", "input": "the pack's manifest.json without ple.rows.pfw", "code": "missing-part"},
                      {"id": "pack-short-shard", "input": "the pack with lm.global.pfw cut short by 16 bytes", "code": "truncated"},
                      {"id": "pack-changed-chunk", "input": "the pack with one byte of lm.global.pfw changed", "result": "verifyModel (full) refuses it; verifyChunk is false for that chunk only"}]}
    doc["packs"] = [make_pack(out)]; print("pack     lma-test-pack/ (%d files)" % len(doc["packs"][0]["files"]))
    if not skip_big:
        rec = make(None, "big", dry=True)
        doc["large"] = [{"id": "big", "label": rec["label"], "generate": "python tools/make_test_models.py DIR --big", "format": "gguf",
                         "files": [{"file": rec["files"][0]["saveAs"], "size": rec["files"][0]["size"], "sha256": rec["files"][0]["sha256"]}], "identity": identity(rec)}]
        print("big      %s %d B (identity only, not written)" % (rec["files"][0]["saveAs"], rec["files"][0]["size"]))
    (out / "vectors.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def main(argv):
    if not argv or argv[0].startswith("-"): raise SystemExit(__doc__)
    if argv[0] == "vectors": return vectors(argv[1], "--skip-big" in argv)
    if argv[0] == "identity": print(json.dumps(identity(make(None, argv[1], dry=True)), indent=1)); return
    if argv[0] == "approve":
        entries = json.loads((Path(argv[1]) / "models.json").read_text(encoding="utf-8"))["models"]
        for c in argv[2:]: approve(c, entries); print("approved", len(entries), "test models in", c)
        return
    out = Path(argv[0]); out.mkdir(parents=True, exist_ok=True)
    ext = argv[argv.index("--ext") + 1] if "--ext" in argv else None
    keys = ["small", "st"] + (["mid"] if "--mid" in argv else []) + (["big"] if "--big" in argv else [])
    mj = out / "models.json"
    try: cur = json.loads(mj.read_text(encoding="utf-8"))["models"]
    except Exception: cur = []
    for k in keys:
        e = make(out, k, ext); cur = [c for c in cur if c.get("key") != k] + [e]
        print("%-6s %-26s %12d bytes  sha256 %s" % (k, e["file"], e["size"], e["sha256"]))
    order = list(PRESETS); cur.sort(key=lambda c: order.index(c["key"]) if c.get("key") in order else 99)
    mj.write_text(json.dumps({"models": cur}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
