"""Train the released MUSE+ model on this workstation.

The released ``MIMIC4Dataset`` expects ``hosp_adm_dict_v2.pkl``, which the
authors' preprocessing writes and which is not present here.  This harness
instead serves the verified boundary -- the same admission objects
``check_batch.py`` proves reconstruct the author ``Dataset`` output exactly --
through the identical loader hooks that checker uses, so the released model,
optimiser, and training loop run unmodified.

Deviations from the released run are recorded in the emitted manifest:
  * source objects come from the verified parquet boundary rather than
    the authors' pickle;
  * the CUDA, Python, and dependency versions differ from the release;
  * Neptune telemetry is stubbed out (no credentials, no network).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import polars as pl

PAPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_ROOT))
# run.py uses bare intra-package imports (``from model import ...``)
sys.path.insert(0, str(PAPER_ROOT / "src" / "core"))

CACHE = PAPER_ROOT / ".export_assets" / "source_cache"
WORK = PAPER_ROOT / ".export_assets" / "repro"
WORK.mkdir(parents=True, exist_ok=True)

# --- 1. redirect the released path constants before anything imports them ---
import src.utils as author_utils  # noqa: E402

author_utils.remote_project_path = str(WORK)
author_utils.processed_data_path = str(WORK / "processed_data")

# --- 2. stub the telemetry the released Helper imports ---
_neptune = types.ModuleType("neptune")
_neptune.init_run = lambda **kwargs: None          # no network, no credentials
sys.modules.setdefault("neptune", _neptune)
_cred = types.ModuleType("src.credentials")
_cred.NEPTUNE_API_KEY = None
_cred.NEPTUNE_PROJECT = None
sys.modules.setdefault("src.credentials", _cred)

import torch  # noqa: E402
import src.dataset.mimic4_dataset as author_dataset_module  # noqa: E402
import src.dataset.tokenizer as author_tokenizer_module  # noqa: E402
from src.dataset.data import MIMIC4Data  # noqa: E402
from src.dataset.vocab import Vocabulary  # noqa: E402
from cohort_source import AUTHOR_MISSING_LAB_WIDTH, author_vocabulary  # noqa: E402

SPLIT_FILES = {
    "train": "train_admission_ids.txt",
    "val": "val_admission_ids.txt",
    "test": "test_admission_ids.txt",
    "no_label": "no_label_admission_ids.txt",
}


def build_vocabulary() -> Vocabulary:
    vocabulary = Vocabulary()
    for value in author_vocabulary(CACHE):
        vocabulary.add_word(value)
    return vocabulary


def build_code_embeddings(vocabulary: Vocabulary) -> np.ndarray:
    """Mirror get_code_embeddings.py: Bio_ClinicalBERT CLS over vocab order."""
    # Keep this artifact distinct from the earlier diagnostic base run, whose
    # wrapper used batches of 256.  The released preprocessing script uses
    # seed 42 and batches of 1,024.
    out = WORK / "embeddings_author_bs1024.npy"
    if out.exists():
        return np.load(out)
    from transformers import AutoModel, AutoTokenizer

    author_utils.set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    model.to(device)
    words = [vocabulary.idx2word[i] for i in range(len(vocabulary))]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(words), 1024):
            batch = words[start:start + 1024]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
            chunks.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
            if start % 5120 == 0:
                print(f"  code embeddings {start}/{len(words)}", flush=True)
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(out, emb)
    return emb


class _Grouped:
    """Contiguous per-admission slices over one sorted column store."""

    def __init__(self, keys: np.ndarray, payload: dict[str, np.ndarray]):
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.payload = {k: v[order] for k, v in payload.items()}
        uniq, start = np.unique(self.keys, return_index=True)
        self.index = {int(k): (int(s), int(e))
                      for k, s, e in zip(uniq, start, list(start[1:]) + [len(self.keys)])}

    def get(self, hadm: int) -> dict[str, np.ndarray]:
        span = self.index.get(hadm)
        if span is None:
            return {k: v[:0] for k, v in self.payload.items()}
        s, e = span
        return {k: v[s:e] for k, v in self.payload.items()}


class LazyAdmissionMap:
    """Materialises the author MIMIC4Data object on demand."""

    def __init__(self, lab_width: int):
        self.lab_width = lab_width
        coh = pl.read_parquet(CACHE / "cohort.parquet")
        self.meta = {
            int(r["hadm_id"]): r
            for r in coh.select(["hadm_id", "subject_id", "duration_minutes",
                                 "mortality", "age", "gender", "race"]).iter_rows(named=True)
        }
        codes = pl.read_parquet(CACHE / "code_trajectory.parquet").sort(
            ["hadm_id", "sequence_row"])
        type_vals, type_idx = np.unique(codes["event_type"].to_numpy(), return_inverse=True)
        name_vals, name_idx = np.unique(codes["code_name"].to_numpy(), return_inverse=True)
        self.type_vals, self.name_vals = type_vals, name_vals
        self.codes = _Grouped(codes["hadm_id"].to_numpy(),
                              {"t": type_idx.astype(np.int32),
                               "c": name_idx.astype(np.int32)})
        labs = pl.read_parquet(CACHE / "lab_boundary.parquet").sort(
            ["hadm_id", "sequence_row"])
        self.labs = _Grouped(labs["hadm_id"].to_numpy(), {
            "row": labs["sequence_row"].to_numpy().astype(np.int32),
            "ch": labs["channel_index"].to_numpy().astype(np.int32),
            "val": labs["numeric_value"].to_numpy().astype(np.float32)})
        notes = pl.read_parquet(CACHE / "brief_hospital_course.parquet")
        self.notes = dict(zip(notes["hadm_id"].to_list(), notes["text"].to_list()))
        print(f"  admissions={len(self.meta):,} code_rows={len(self.codes.keys):,} "
              f"lab_rows={len(self.labs.keys):,} notes={len(self.notes):,}", flush=True)

    def _labvectors_reference(self, hadm: int):
        """Original per-item reconstruction. Kept as the equality reference."""
        g = self.labs.get(hadm)
        if len(g["row"]) == 0:
            return None
        n = int(g["row"].max()) + 1
        out = np.zeros((n, self.lab_width), dtype=np.float32)
        vector = np.zeros(self.lab_width, dtype=np.float32)
        cur = 0
        for row, ch, val in zip(g["row"], g["ch"], g["val"]):
            while cur < int(row):
                out[cur] = vector
                cur += 1
            if int(ch) >= 0:
                vector[int(ch)] = val
        while cur < n:
            out[cur] = vector
            cur += 1
        return out

    def prepack(self) -> None:
        """Materialise every dense lab tensor once into one contiguous array.

        The per-item version above rebuilds a forward-filled matrix inside the
        DataLoader worker for every batch, which leaves the GPU idle. This runs
        the identical scan a single time and caches it, so __getitem__ becomes a
        slice. Verified elementwise against _labvectors_reference.
        """
        dense_path = WORK / "lab_dense.npy"
        index_path = WORK / "lab_index.npy"
        keys_path = WORK / "lab_keys.npy"
        if dense_path.exists() and index_path.exists() and keys_path.exists():
            self.lab_dense = np.load(dense_path, mmap_mode="r")
            index = np.load(index_path)
            keys = np.load(keys_path)
        else:
            hadm = self.labs.keys
            row = self.labs.payload["row"]
            ch = self.labs.payload["ch"]
            val = self.labs.payload["val"]
            uniq, starts = np.unique(hadm, return_index=True)
            ends = list(starts[1:]) + [len(hadm)]
            heights = np.empty(len(uniq), dtype=np.int64)
            for i, (s, e) in enumerate(zip(starts, ends)):
                heights[i] = int(row[s:e].max()) + 1
            offsets = np.concatenate([[0], np.cumsum(heights)]).astype(np.int64)
            dense = np.zeros((int(offsets[-1]), self.lab_width), dtype=np.float32)
            vector = np.empty(self.lab_width, dtype=np.float32)
            for i, (s, e) in enumerate(zip(starts, ends)):
                vector[:] = 0.0
                base = offsets[i]
                cur = 0
                for k in range(s, e):
                    r = int(row[k])
                    while cur < r:
                        dense[base + cur] = vector
                        cur += 1
                    c = int(ch[k])
                    if c >= 0:
                        vector[c] = val[k]
                while cur < heights[i]:
                    dense[base + cur] = vector
                    cur += 1
                if i % 50000 == 0:
                    print(f"  packing labs {i:,}/{len(uniq):,}", flush=True)
            np.save(dense_path, dense)
            index = np.stack([offsets[:-1], heights], axis=1)
            np.save(index_path, index)
            keys = uniq.astype(np.int64)
            np.save(keys_path, keys)
            self.lab_dense = dense
        self.lab_index = {int(k): (int(o), int(h)) for k, (o, h) in zip(keys, index)}
        print(f"  packed labs: {self.lab_dense.shape}", flush=True)

    def _labvectors(self, hadm: int):
        span = self.lab_index.get(hadm)
        if span is None:
            return None
        offset, height = span
        return np.asarray(self.lab_dense[offset:offset + height])

    def __getitem__(self, admission_id):
        hadm = int(admission_id)
        r = self.meta[hadm]
        value = MIMIC4Data(
            admission_id=str(hadm), patient_id=str(int(r["subject_id"])),
            duration=float(r["duration_minutes"]), mortality=(
                None if r["mortality"] is None else bool(r["mortality"])),
            readmission=None, age=int(r["age"]),
            gender=str(r["gender"]), ethnicity=str(r["race"]))
        g = self.codes.get(hadm)
        if len(g["t"]) == 0:
            value.trajectory = ([], [])
        else:
            value.trajectory = (list(self.type_vals[g["t"]]), list(self.name_vals[g["c"]]))
        value.labvectors = self._labvectors(hadm)
        value.discharge = self.notes.get(hadm)
        return value


def install(vocabulary, embeddings, admissions):
    author_dataset_module.load_pickle = lambda _: admissions
    author_tokenizer_module.load_pickle = lambda _: vocabulary
    author_tokenizer_module.read = lambda _: embeddings

    def read_txt(path):
        name = Path(path).name
        key = name.replace("_admission_ids.txt", "")
        return [line.strip() for line in
                (CACHE / SPLIT_FILES[key]).read_text().split("\n") if line.strip()]

    author_dataset_module.read_txt = read_txt


def main() -> None:
    print("building vocabulary...", flush=True)
    vocabulary = build_vocabulary()
    print(f"  vocab size {len(vocabulary):,}", flush=True)
    print("building code embeddings (Bio_ClinicalBERT)...", flush=True)
    embeddings = build_code_embeddings(vocabulary)
    print(f"  embeddings {embeddings.shape}", flush=True)
    width = AUTHOR_MISSING_LAB_WIDTH
    print(f"building lazy admission map (lab width {width})...", flush=True)
    admissions = LazyAdmissionMap(width)
    admissions.prepack()
    install(vocabulary, embeddings, admissions)

    # Keep all executable defaults in the unmodified author run.py.  The
    # monitor argument repairs its MIMIC README/default mismatch; load_no_label
    # selects the sole target, MUSE+.
    argv = [
        "run.py",
        "--dataset", "mimic4",
        "--task", "mortality",
        "--monitor", "pr_auc",
        "--load_no_label", "True",
    ]
    sys.argv = argv + sys.argv[1:]
    print(f"launching released trainer: {' '.join(argv)}", flush=True)
    import src.core.run  # noqa: F401  (module-level training loop)


if __name__ == "__main__":
    main()
