"""Decode REMed/snub tokenized events into plain MEDS codes and values.

The hashed-vocabulary export collapsed every distinct serialized event into
its own opaque code (see the `remed-plain-timeline` project note). This
module inverts that: each [3,128] GenHPF event decodes deterministically —
type ids segment table name (1), column names (2), and values (3);
WordPiece joins through the pinned Bio_ClinicalBERT vocabulary; numeric
strings collapse their per-digit spacing — and per-table rules emit plain
`MIMICIV//<table>//<identity>` codes with real numeric values.

Parity contract (user-directed): the event set is exactly the model's own
h5 stream, so neither side can see an event the other lacks. Per event we
retain the identity column and every primary numeric the model sees;
a categorical value is embedded into an occurrence code only when that
exact (item, value) pair appears in enough TRAIN stays (per-pair support,
never per-item). Free-text fields (comments, chief complaints) are dropped
and disclosed.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PAD_TYPE = 0
TABLE_TYPE = 1
COLUMN_TYPE = 2
VALUE_TYPE = 3
TIMEINT_TYPE = 4
SPECIAL_TYPES = (5, 6)

EVENT_PAYLOAD_PREFIX = "SNUB_HI_EVENT/"

# A categorical value is embedded into its item's code only when at least
# this many TRAIN stays carry that exact (item, value) pair — decided
# per pair, never per item, so a mixed item keeps its common picklist
# values while one-off free text falls back to identity-only. Ten matches
# the protocol's min_child_samples floor: a rarer code could never support
# a split anyway.
MINIMUM_VALUE_SUPPORT_STAYS = 10
MAXIMUM_VALUE_LENGTH = 48
# The plain vocabulary must stay orders of magnitude below the event count;
# a breach means value-embedding leaked unbounded identities again.
MAXIMUM_PLAIN_CODES = 150_000

_NUMBER_SPACING = re.compile(r"(?<=[0-9.\-]) (?=[0-9.])")
_JUNK_VALUE = re.compile(r"^[\s\-_.]*$")


def load_vocabulary(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.rstrip("\n") for line in handle]


def decode_payload(description: str) -> np.ndarray:
    if not description.startswith(EVENT_PAYLOAD_PREFIX):
        raise ValueError("not a canonical snub event payload")
    encoded = description[len(EVENT_PAYLOAD_PREFIX) :]
    raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    width = raw[0]
    if width > 128 or len(raw) != 1 + 3 * width * 2:
        raise ValueError("malformed snub event payload")
    return np.frombuffer(raw[1:], dtype="<i2").reshape(3, width)


def _join_wordpieces(vocabulary: list[str], token_ids: list[int]) -> str:
    text = ""
    for token_id in token_ids:
        piece = vocabulary[token_id]
        if piece.startswith("##"):
            text += piece[2:]
        else:
            text += (" " if text else "") + piece
    return text


def segment_event(
    vocabulary: list[str], event: np.ndarray
) -> tuple[str, list[tuple[str, str]]]:
    """Return (table_name, [(column, value), ...]) for one decoded event."""
    runs: list[tuple[int, list[int]]] = []
    run_type: int | None = None
    for token, token_type in zip(event[0], event[1]):
        token_type = int(token_type)
        if token_type == PAD_TYPE:
            if int(token) != 0:
                raise ValueError("pad-typed token is not the [PAD] id")
            run_type = None
            continue
        if token_type in SPECIAL_TYPES or token_type == TIMEINT_TYPE:
            run_type = None
            continue
        if token_type != run_type:
            runs.append((token_type, []))
            run_type = token_type
        runs[-1][1].append(int(token))
    if not runs or runs[0][0] != TABLE_TYPE:
        raise ValueError("event does not start with a table-name run")
    table = _join_wordpieces(vocabulary, runs[0][1])
    fields: list[tuple[str, str]] = []
    index = 1
    while index < len(runs):
        run_kind, tokens = runs[index]
        if run_kind != COLUMN_TYPE:
            raise ValueError(f"expected a column run in table {table!r}")
        column = _join_wordpieces(vocabulary, tokens).replace(" _ ", "_")
        value = ""
        if index + 1 < len(runs) and runs[index + 1][0] == VALUE_TYPE:
            value = _join_wordpieces(vocabulary, runs[index + 1][1])
            index += 1
        fields.append((column, value))
        index += 1
    return table, fields


def collapse_number_spacing(value: str) -> str:
    return _NUMBER_SPACING.sub("", value)


def normalize_value(value: str) -> str:
    """Collapse digit spacing; masked/empty markers become empty."""
    collapsed = collapse_number_spacing(value.strip())
    if _JUNK_VALUE.match(collapsed):
        return ""
    return collapsed


def clamp_duration(
    numeric: float, minute: float, prediction_minutes: float
) -> float:
    """Truncate a start-anchored duration to what was observable at
    prediction time: min(duration, minutes remaining in the window)."""
    remaining = prediction_minutes - minute
    return numeric if numeric <= remaining else remaining


def parse_numeric(value: str) -> float | None:
    collapsed = collapse_number_spacing(value.strip())
    if not collapsed:
        return None
    try:
        parsed = float(collapsed)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


# Per-table rules: identity column, numeric emissions (column -> code
# suffix), flag-style occurrence columns (column, value -> suffix), and
# categorical-value column (embedded subject to the cardinality policy).
@dataclass(frozen=True)
class TableRule:
    prefix: str
    identity_columns: tuple[str, ...]
    numeric_columns: tuple[tuple[str, str], ...] = ()
    flag_columns: tuple[tuple[str, str, str], ...] = ()
    categorical_value_column: str | None = None
    per_column_numeric: bool = False


TABLE_RULES: dict[str, TableRule] = {
    "labevents": TableRule(
        prefix="LAB",
        identity_columns=("itemid",),
        numeric_columns=(("valuenum", ""),),
        flag_columns=(("flag", "abnormal", "abnormal"),),
        categorical_value_column="value",
    ),
    "chartevents": TableRule(
        prefix="CHART",
        identity_columns=("itemid",),
        numeric_columns=(("valuenum", ""),),
        flag_columns=(("warning", "1", "warning"),),
        categorical_value_column="value",
    ),
    "inputevents": TableRule(
        prefix="INPUT",
        identity_columns=("itemid",),
        numeric_columns=(("amount", ""), ("rate", "rate")),
    ),
    "outputevents": TableRule(
        prefix="OUT",
        identity_columns=("itemid",),
        numeric_columns=(("value", ""),),
    ),
    "procedureevents": TableRule(
        prefix="PROC",
        identity_columns=("itemid",),
        numeric_columns=(("value", ""),),
    ),
    "prescriptions": TableRule(
        prefix="RX",
        identity_columns=("drug",),
        numeric_columns=(("dose_val_rx", "dose"),),
    ),
    "microbiologyevents": TableRule(
        prefix="MICRO",
        identity_columns=("test_name",),
        flag_columns=(),
        categorical_value_column=None,
    ),
    "diagnosis": TableRule(
        prefix="EDDX",
        identity_columns=("icd_title",),
    ),
    "medrecon": TableRule(
        prefix="MEDREC",
        identity_columns=("name",),
    ),
    "pyxis": TableRule(
        prefix="PYXIS",
        identity_columns=("name",),
    ),
    "triage": TableRule(
        prefix="TRIAGE",
        identity_columns=(),
        per_column_numeric=True,
    ),
    "vitalsign": TableRule(
        prefix="EDVITAL",
        identity_columns=(),
        per_column_numeric=True,
    ),
}

# Free-text or constant-per-item fields the model sees but the plain
# timeline drops; disclosed in the manifest.
DISCLOSED_DROPPED_COLUMNS = {
    "labevents": ("valueuom", "ref_range_lower", "ref_range_upper",
                  "priority", "comments"),
    "chartevents": ("valueuom",),
    "inputevents": ("amountuom", "rateuom", "ordercategoryname",
                    "secondaryordercategoryname",
                    "ordercomponenttypedescription",
                    "ordercategorydescription", "patientweight",
                    "totalamount", "totalamountuom", "isopenbag",
                    "originalamount", "originalrate"),
    "outputevents": ("valueuom",),
    "procedureevents": ("valueuom", "location", "locationcategory",
                        "ordercategoryname", "ordercategorydescription",
                        "patientweight", "isopenbag", "originalamount",
                        "originalrate"),
    "prescriptions": ("drug_type", "prod_strength", "dose_unit_rx",
                      "form_val_disp", "form_unit_disp",
                      "doses_per_24_hrs", "route"),
    "microbiologyevents": ("org_name", "ab_name", "interpretation",
                           "test_seq", "isolate_num", "dilution_text",
                           "dilution_comparison", "dilution_value",
                           "comments"),
    "diagnosis": ("seq_num", "icd_version"),
    "medrecon": ("etcdescription",),
    "triage": ("chiefcomplaint",),
    "vitalsign": (),
    "pyxis": (),
}

# Microbiology emissions are structural rather than rule-table-driven: the
# model sees specimen, organism, and per-antibiotic susceptibility, all of
# which are bounded categorical signal.
_MICRO_TABLE = "microbiologyevents"


@dataclass
class DecodeStatistics:
    events: int = 0
    emissions: int = 0
    numeric_emissions: int = 0
    occurrence_emissions: int = 0
    missing_identity: int = 0
    unparseable_numeric: int = 0
    retained_value_pairs: int = 0
    rejected_value_pairs: int = 0
    withheld_future_values: int = 0
    tables: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "events": self.events,
            "emissions": self.emissions,
            "numeric_emissions": self.numeric_emissions,
            "occurrence_emissions": self.occurrence_emissions,
            "missing_identity": self.missing_identity,
            "unparseable_numeric": self.unparseable_numeric,
            "retained_value_pairs": self.retained_value_pairs,
            "rejected_value_pairs": self.rejected_value_pairs,
            "withheld_future_values": self.withheld_future_values,
            "events_by_table": dict(sorted(self.tables.items())),
        }


class PlainEventDecoder:
    """Two-pass decoder with per-(item, value) support-based embedding.

    Pass 1 (`survey_sample`, train stays only): count, per candidate
    (code-base, value) pair, how many stays carry it — deduplicated within
    a stay. Pass 2 (`emissions`): a categorical value is embedded into the
    code only when its pair reached `MINIMUM_VALUE_SUPPORT_STAYS`; decided
    for each pair individually (heart rhythm//af stands on its own),
    never for the whole item. Parsing is memoized on the raw event bytes,
    so each of the ~2.6M distinct events decodes once.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary
        self._parse_cache: dict[bytes, tuple[str, tuple[tuple[str, str], ...]]] = {}
        self._support: dict[tuple[str, str], int] = {}
        self._retained: set[tuple[str, str]] | None = None
        self._emission_cache: dict[bytes, list[tuple[str, float | None]]] = {}
        self.statistics = DecodeStatistics()

    # -- shared parsing ---------------------------------------------------
    def _parse(self, event: np.ndarray) -> tuple[str, tuple[tuple[str, str], ...]]:
        key = event.tobytes()
        cached = self._parse_cache.get(key)
        if cached is None:
            table, fields = segment_event(self.vocabulary, event)
            cached = (table, tuple(fields))
            self._parse_cache[key] = cached
        return cached

    def parse_description(self, description: str):
        return self._parse(decode_payload(description))

    def _candidates(
        self, table: str, fields: tuple[tuple[str, str], ...]
    ) -> list[tuple[str, str]]:
        """(code-base, normalized value) pairs eligible for embedding."""
        rule = TABLE_RULES.get(table)
        if rule is None:
            raise ValueError(f"undeclared snub table {table!r}")
        mapping = dict(fields)
        out: list[tuple[str, str]] = []
        if rule.per_column_numeric:
            for column, value in fields:
                if column in DISCLOSED_DROPPED_COLUMNS.get(table, ()):
                    continue
                if parse_numeric(value) is not None:
                    continue
                normalized = normalize_value(value)
                if normalized and len(normalized) <= MAXIMUM_VALUE_LENGTH:
                    out.append((f"MIMICIV//{rule.prefix}//{column}", normalized))
            return out
        if rule.categorical_value_column is None:
            return out
        numeric_column = (
            rule.numeric_columns[0][0] if rule.numeric_columns else None
        )
        if numeric_column and parse_numeric(mapping.get(numeric_column, "")) is not None:
            return out
        identity = " ".join(
            mapping.get(column, "") for column in rule.identity_columns
        ).strip()
        value = normalize_value(mapping.get(rule.categorical_value_column, ""))
        if identity and value and len(value) <= MAXIMUM_VALUE_LENGTH:
            out.append((f"MIMICIV//{rule.prefix}//{identity}", value))
        return out

    # -- pass 1 -----------------------------------------------------------
    def survey_sample(self, events: "list[np.ndarray] | np.ndarray") -> None:
        """Count one stay's candidate pairs (deduplicated within the stay)."""
        stay_pairs: set[tuple[str, str]] = set()
        for event in events:
            table, fields = self._parse(np.asarray(event, dtype=np.int16))
            stay_pairs.update(self._candidates(table, fields))
        for pair in stay_pairs:
            self._support[pair] = self._support.get(pair, 0) + 1

    def finalize_survey(self) -> None:
        self._retained = {
            pair
            for pair, stays in self._support.items()
            if stays >= MINIMUM_VALUE_SUPPORT_STAYS
        }
        self.statistics.retained_value_pairs = len(self._retained)
        self.statistics.rejected_value_pairs = (
            len(self._support) - len(self._retained)
        )

    # -- pass 2 -----------------------------------------------------------
    def emissions(
        self, event: np.ndarray
    ) -> list[tuple[str, float | None, bool]]:
        if self._retained is None:
            raise RuntimeError("finalize_survey must run before emissions")
        event = np.asarray(event, dtype=np.int16)
        key = event.tobytes()
        out = self._emission_cache.get(key)
        if out is None:
            out = self._build_emissions(*self._parse(event))
            self._emission_cache[key] = out
        statistics = self.statistics
        statistics.events += 1
        statistics.emissions += len(out)
        table = self._parse_cache[key][0]
        statistics.tables[table] = statistics.tables.get(table, 0) + 1
        for _, numeric, _clamp in out:
            if numeric is None:
                statistics.occurrence_emissions += 1
            else:
                statistics.numeric_emissions += 1
        return out

    def emissions_for_description(self, description: str):
        return self.emissions(decode_payload(description))

    def _build_emissions(
        self, table: str, fields: tuple[tuple[str, str], ...]
    ) -> list[tuple[str, float | None, bool]]:
        statistics = self.statistics
        rule = TABLE_RULES.get(table)
        if rule is None:
            raise ValueError(f"undeclared snub table {table!r}")
        mapping = dict(fields)
        retained = self._retained
        # Interval-completion defect (docs/remed_plain_timeline.md): the
        # source anchors procedure durations at start time, so a
        # minute-denominated value can reach past the prediction cutoff.
        # Such emissions are flagged for clamping at export time.
        clamp = (
            table == "procedureevents"
            and mapping.get("valueuom", "") == "min"
        )
        out: list[tuple[str, float | None, bool]] = []

        if table == _MICRO_TABLE:
            out.extend(self._micro_emissions(mapping))
        elif rule.per_column_numeric:
            for column, value in fields:
                if column in DISCLOSED_DROPPED_COLUMNS.get(table, ()):
                    continue
                code = f"MIMICIV//{rule.prefix}//{column}"
                numeric = parse_numeric(value)
                if numeric is not None:
                    out.append((code, numeric, False))
                    continue
                normalized = normalize_value(value)
                if (
                    normalized
                    and len(normalized) <= MAXIMUM_VALUE_LENGTH
                    and (code, normalized) in retained
                ):
                    out.append((f"{code}//{normalized}", None, False))
                elif normalized:
                    # The bare code carries numerics; a non-numeric,
                    # unretained value must not mix occurrence rows into it.
                    out.append((f"{code}//other", None, False))
        else:
            identity = " ".join(
                mapping.get(column, "") for column in rule.identity_columns
            ).strip()
            if not identity:
                identity = "unlabeled"
                statistics.missing_identity += 1
            base = f"MIMICIV//{rule.prefix}//{identity}"
            emitted_numeric = False
            # Interval-completion defect, inputevents flavor: a continuous
            # infusion's `amount` is the total over an interval that can run
            # past prediction time (amount/rate reconstructs the future
            # duration), so it is withheld whenever a positive rate marks
            # the row as continuous; the instantaneous rate is retained.
            continuous_infusion = (
                table == "inputevents"
                and (parse_numeric(mapping.get("rate", "")) or 0.0) > 0.0
            )
            for column, suffix in rule.numeric_columns:
                raw = mapping.get(column, "")
                if not raw:
                    continue
                if continuous_infusion and column == "amount":
                    statistics.withheld_future_values += 1
                    continue
                numeric = parse_numeric(raw)
                if numeric is None:
                    statistics.unparseable_numeric += 1
                    continue
                code = base if not suffix else f"{base}//{suffix}"
                out.append((code, numeric, clamp))
                emitted_numeric = True
            if not emitted_numeric:
                value = normalize_value(
                    mapping.get(rule.categorical_value_column or "", "")
                )
                if (
                    value
                    and len(value) <= MAXIMUM_VALUE_LENGTH
                    and (base, value) in retained
                ):
                    out.append((f"{base}//{value}", None, False))
                elif any(suffix == "" for _, suffix in rule.numeric_columns):
                    # The bare code carries numerics for this table; keep it
                    # pure by routing value-less/unretained rows aside.
                    out.append((f"{base}//other", None, False))
                else:
                    out.append((base, None, False))
            for column, expected, suffix in rule.flag_columns:
                if mapping.get(column, "") == expected:
                    out.append((f"{base}//{suffix}", None, False))

        if not out:
            out.append((f"MIMICIV//{rule.prefix}//unlabeled", None, False))
            statistics.missing_identity += 1
        return out

    def _micro_emissions(
        self, mapping: dict[str, str]
    ) -> list[tuple[str, float | None, bool]]:
        """Only the specimen/test is knowable at collection time.

        The source anchors culture results (organism, antibiotic
        susceptibility) at the specimen's chart time although they become
        known days later — the result-bearing-fields defect. The model's
        input carries them; the baseline withholds them.
        """
        test = mapping.get("test_name", "").strip() or "unlabeled"
        if mapping.get("org_name", "").strip() or (
            mapping.get("ab_name", "").strip()
            and mapping.get("interpretation", "").strip()
        ):
            self.statistics.withheld_future_values += 1
        return [(f"MIMICIV//MICRO//{test}", None, False)]
