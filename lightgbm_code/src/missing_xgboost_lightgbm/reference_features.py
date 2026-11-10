"""Slow, literal Python oracle for the production Rust featurizer.

This module favors obvious nested loops over performance. It is used by tests,
not model runs, so that the optimized Rust implementation has an independent
and readable correctness reference.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import FeatureMatrix


SUMMARY_NAMES = (
    "count",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "latest_observed",
    "delta",
    # Timing is measured backwards from prediction_time, never forwards from
    # observation_start: only the prediction-relative anchor is comparable
    # across samples whose windows differ in length.
    "first_recency",
    "recency",
)
OCCURRENCE_SUMMARY_NAMES = (
    "count",
    "first_recency",
    "recency",
)
def _inside(time: float, start: float, stop: float, hours: float) -> bool:
    return time >= start and (time <= stop if stop == hours else time < stop)


def _numeric_statistics(
    series: list[tuple[float, int, int, float]],
    start: float,
    stop: float,
    hours: float,
) -> np.ndarray:
    selected = [row for row in series if _inside(row[0], start, stop, hours)]
    if not selected:
        return np.asarray((0.0,) + (np.nan,) * 9, dtype=np.float32)
    times = np.asarray([row[0] for row in selected], dtype=np.float64)
    values = np.asarray([row[3] for row in selected], dtype=np.float64)
    first = float(values[0])
    latest = float(values[-1])
    latest_time = float(times[-1])
    return np.asarray(
        (
            len(selected),
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            latest,
            latest - first,
            hours - times[0],
            hours - latest_time,
        ),
        dtype=np.float32,
    )


def _occurrence_statistics(
    series: list[tuple[float, int, int, float]],
    start: float,
    stop: float,
    hours: float,
) -> np.ndarray:
    times = [row[0] for row in series if _inside(row[0], start, stop, hours)]
    if not times:
        return np.zeros(len(OCCURRENCE_SUMMARY_NAMES), dtype=np.float32)
    return np.asarray(
        (
            len(times),
            hours - times[0],
            hours - times[-1],
        ),
        dtype=np.float32,
    )


def _groups(
    events: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    codes: tuple[str, ...],
    occurrence_codes: tuple[str, ...],
    sample_window_hours: np.ndarray,
) -> dict[tuple[int, int], list[tuple[float, int, int, float]]]:
    required = {"sample_id", "time_hours", "code", "value", "source_row"}
    if set(events) != required:
        raise ValueError(f"reference events must contain exactly {sorted(required)}")
    lengths = {len(np.asarray(value)) for value in events.values()}
    if len(lengths) != 1:
        raise ValueError("reference event arrays have inconsistent lengths")
    sample_lookup = {int(value): index for index, value in enumerate(sample_ids)}
    code_lookup = {value: index for index, value in enumerate(codes)}
    occurrences = set(occurrence_codes)
    grouped = defaultdict(list)
    for input_row, (sample, time, code, value, source_row) in enumerate(
        zip(
            events["sample_id"],
            events["time_hours"],
            events["code"],
            events["value"],
            events["source_row"],
            strict=True,
        )
    ):
        sample_index = sample_lookup.get(int(sample))
        code = str(code)
        code_index = code_lookup.get(code)
        if sample_index is None or code_index is None:
            continue
        time = float(time)
        value = float(value)
        if (
            not np.isfinite(time)
            or not 0.0 <= time <= sample_window_hours[sample_index]
        ):
            raise ValueError("selected event time is outside its sample window")
        if code in occurrences:
            if not np.isnan(value):
                raise ValueError("occurrence-only event must have a missing value")
        elif not np.isfinite(value):
            raise ValueError("numeric event value must be finite")
        grouped[(sample_index, code_index)].append(
            (time, int(source_row), input_row, value)
        )
    for series in grouped.values():
        series.sort(key=lambda row: (row[0], row[1], row[2]))
    return grouped


def _summary_names(
    variables: tuple[str, ...], statistics: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        f"{variable}__full__{statistic}"
        for variable in variables
        for statistic in statistics
    )


# Representations built from a small set of per-sample segments. Both share one
# mechanic and differ only in how their boundaries are derived:
#
#   visit_halves   full / earlier / recent, split at a single cohort-wide
#                  duration D before prediction time. D is absolute rather than
#                  a fraction of each sample's window so that "recent" denotes
#                  the same elapsed span for every patient; a per-sample
#                  fraction would mean 12 hours for a 24-hour stay and 100 for
#                  a 200-hour one, which is the window-length dependence that
#                  prediction-relative anchoring exists to remove.
#   visit_buckets  all / prior / current, split at the sample's last event
#                  time. For a task whose events carry one timestamp per visit,
#                  that boundary is exactly the current visit.
SEGMENTS_BY_REPRESENTATION = {
    "visit_halves": ("full", "earlier", "recent"),
    "visit_buckets": ("all", "prior", "current"),
}

# Every segmented representation a profile may search (a tuple; empty for none).
SEGMENTED_BY_PROFILE = {
    "single_visit_timed": ("visit_halves",),
    "multi_visit_untimed": ("visit_buckets",),
}
# A multi_visit_untimed source model is shown visit *order* and never elapsed
# time, so the two prediction-relative timing statistics would hand LightGBM a
# signal the published method provably cannot see. Every remaining statistic
# summarizes values, not instants. For a task whose codes are all
# occurrence-only, this leaves exactly `count`.
TIMING_STATISTICS = ("first_recency", "recency")
TIMED_PROFILES = frozenset(
    profile for profile in SEGMENTED_BY_PROFILE if profile != "multi_visit_untimed"
)
# Per-sample totals, identical in every representation. `window_hours` is the
# elapsed observation width at prediction time, never the eventual stay length,
# and it is withheld from untimed profiles for the same reason as above.
AGGREGATE_NAMES = (
    "__sample__event_count",
    "__sample__code_count",
    "__sample__distinct_time_count",
)
TIMED_AGGREGATE_NAMES = AGGREGATE_NAMES + ("__sample__window_hours",)


def statistic_names(timed: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The numeric and occurrence statistic lists for one temporal profile."""
    if timed:
        return SUMMARY_NAMES, OCCURRENCE_SUMMARY_NAMES
    return (
        tuple(name for name in SUMMARY_NAMES if name not in TIMING_STATISTICS),
        tuple(
            name
            for name in OCCURRENCE_SUMMARY_NAMES
            if name not in TIMING_STATISTICS
        ),
    )


def aggregate_names(timed: bool) -> tuple[str, ...]:
    return TIMED_AGGREGATE_NAMES if timed else AGGREGATE_NAMES


def _statistic_positions(timed: bool) -> tuple[np.ndarray, np.ndarray]:
    numeric, occurrence = statistic_names(timed)
    return (
        np.asarray([SUMMARY_NAMES.index(name) for name in numeric], dtype=np.intp),
        np.asarray(
            [OCCURRENCE_SUMMARY_NAMES.index(name) for name in occurrence],
            dtype=np.intp,
        ),
    )


def cohort_halves_split_hours(sample_window_hours) -> float:
    """Dataset-wide split for `visit_halves`: half the cohort median width."""
    widths = np.asarray(sample_window_hours, dtype=np.float64)
    if widths.size == 0:
        raise ValueError("cohort split needs at least one sample window")
    return 0.5 * float(np.median(widths))


def _segment_bounds(
    representation: str,
    sample_hours: float,
    last_event_hours: float,
    split_hours: float,
) -> tuple[tuple[str, float, float], ...]:
    """(name, start, stop) per segment."""
    if representation == "visit_halves":
        cut = max(0.0, sample_hours - split_hours)
        return (
            ("full", 0.0, sample_hours),
            ("earlier", 0.0, cut),
            ("recent", cut, sample_hours),
        )
    if representation == "visit_buckets":
        return (
            ("all", 0.0, sample_hours),
            ("prior", 0.0, last_event_hours),
            ("current", last_event_hours, sample_hours),
        )
    raise ValueError(f"unknown segmented representation {representation!r}")


def build_segment_representation(
    groups: dict,
    rows: int,
    numeric_variables: tuple[str, ...],
    occurrence_variables: tuple[str, ...],
    numeric_global: list[int],
    occurrence_global: list[int],
    sample_window_hours: np.ndarray,
    last_event_hours: np.ndarray,
    representation: str,
    split_hours: float,
    timed: bool,
) -> FeatureMatrix:
    """Statistics over the representation's segments, literal loops throughout."""
    segments = SEGMENTS_BY_REPRESENTATION[representation]
    count = len(segments)
    numeric_statistics, occurrence_statistics = statistic_names(timed)
    numeric_kept, occurrence_kept = _statistic_positions(timed)
    numeric_width = len(numeric_statistics)
    occurrence_width = len(occurrence_statistics)
    numeric = np.full(
        (rows, len(numeric_variables) * count * numeric_width), np.nan, np.float32
    )
    occurrence = np.full(
        (rows, len(occurrence_variables) * count * occurrence_width),
        np.nan,
        np.float32,
    )
    for sample in range(rows):
        hours = float(sample_window_hours[sample])
        bounds = _segment_bounds(
            representation, hours, float(last_event_hours[sample]), split_hours
        )
        for position, global_code in enumerate(numeric_global):
            series = groups.get((sample, global_code), [])
            for index, (_, start, stop) in enumerate(bounds):
                base = (position * count + index) * numeric_width
                numeric[sample, base : base + numeric_width] = _numeric_statistics(
                    series, start, stop, hours
                )[numeric_kept]
        for position, global_code in enumerate(occurrence_global):
            series = groups.get((sample, global_code), [])
            for index, (_, start, stop) in enumerate(bounds):
                base = (position * count + index) * occurrence_width
                occurrence[sample, base : base + occurrence_width] = (
                    _occurrence_statistics(series, start, stop, hours)[occurrence_kept]
                )
    names = tuple(
        f"{variable}__{segment}__{statistic}"
        for variable in numeric_variables
        for segment in segments
        for statistic in numeric_statistics
    ) + tuple(
        f"{variable}__{segment}__{statistic}"
        for variable in occurrence_variables
        for segment in segments
        for statistic in occurrence_statistics
    )
    return FeatureMatrix(np.concatenate([numeric, occurrence], axis=1), names)


def _sample_aggregates(
    groups: dict,
    rows: int,
    sample_window_hours: np.ndarray,
    timed: bool,
) -> np.ndarray:
    """Per-sample totals: events, distinct codes, distinct instants, width."""
    width = len(aggregate_names(timed))
    values = np.zeros((rows, width), dtype=np.float32)
    times: list[set[float]] = [set() for _ in range(rows)]
    for (sample, _code), series in groups.items():
        values[sample, 0] += len(series)
        values[sample, 1] += 1.0
        times[sample].update(row[0] for row in series)
    for sample in range(rows):
        values[sample, 2] = len(times[sample])
        if timed:
            values[sample, 3] = sample_window_hours[sample]
    return values


def build_reference_event_representations(
    events: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    codes: tuple[str, ...],
    observation_window_hours: float,
    *,
    occurrence_codes: tuple[str, ...] = (),
    sample_window_hours: np.ndarray | None = None,
    temporal_profile: str = "single_visit_timed",
    halves_split_hours: float | None = None,
) -> dict[str, FeatureMatrix]:
    """Build every feature view with a deliberately literal Python algorithm."""
    sample_ids = np.asarray(sample_ids)
    codes = tuple(codes)
    occurrence_codes = tuple(occurrence_codes)
    hours = float(observation_window_hours)
    if not np.isfinite(hours) or hours <= 0.0:
        raise ValueError("observation_window_hours must be positive and finite")
    if temporal_profile not in SEGMENTED_BY_PROFILE:
        raise ValueError(f"unknown temporal_profile {temporal_profile!r}")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("codes must be nonempty and unique")
    if len(set(occurrence_codes)) != len(occurrence_codes) or not set(
        occurrence_codes
    ).issubset(codes):
        raise ValueError("occurrence_codes must be unique declared codes")
    if sample_window_hours is None:
        sample_window_hours = np.full(len(sample_ids), hours, dtype=np.float64)
    else:
        sample_window_hours = np.asarray(sample_window_hours, dtype=np.float64)
        if sample_window_hours.shape != (len(sample_ids),):
            raise ValueError("sample_window_hours must have one value per sample")
        if np.any(~np.isfinite(sample_window_hours)) or np.any(
            (sample_window_hours <= 0.0) | (sample_window_hours > hours)
        ):
            raise ValueError(
                "sample_window_hours must be positive and no wider than the task window"
            )

    groups = _groups(
        events, sample_ids, codes, occurrence_codes, sample_window_hours
    )
    occurrence_set = set(occurrence_codes)
    numeric_variables = tuple(code for code in codes if code not in occurrence_set)
    occurrence_variables = tuple(code for code in codes if code in occurrence_set)
    numeric_global = [codes.index(code) for code in numeric_variables]
    occurrence_global = [codes.index(code) for code in occurrence_variables]
    rows = len(sample_ids)

    timed = temporal_profile in TIMED_PROFILES
    numeric_statistics, occurrence_statistics = statistic_names(timed)
    numeric_kept, occurrence_kept = _statistic_positions(timed)
    numeric_width = len(numeric_statistics)
    occurrence_width = len(occurrence_statistics)
    aggregates = _sample_aggregates(groups, rows, sample_window_hours, timed)

    numeric_summary = np.full(
        (rows, len(numeric_variables) * numeric_width), np.nan, np.float32
    )
    if numeric_variables:
        # `count` is the only statistic defined for an unobserved code.
        numeric_summary.reshape(rows, len(numeric_variables), -1)[:, :, 0] = 0.0
    for variable, global_code in enumerate(numeric_global):
        for sample in range(rows):
            series = groups.get((sample, global_code), [])
            sample_hours = float(sample_window_hours[sample])
            numeric_summary[
                sample, variable * numeric_width : (variable + 1) * numeric_width
            ] = _numeric_statistics(series, 0.0, sample_hours, sample_hours)[
                numeric_kept
            ]

    occurrence_summary = np.zeros(
        (rows, len(occurrence_variables) * occurrence_width), np.float32
    )
    for variable, global_code in enumerate(occurrence_global):
        for sample in range(rows):
            series = groups.get((sample, global_code), [])
            sample_hours = float(sample_window_hours[sample])
            occurrence_summary[
                sample,
                variable * occurrence_width : (variable + 1) * occurrence_width,
            ] = _occurrence_statistics(series, 0.0, sample_hours, sample_hours)[
                occurrence_kept
            ]

    built = {
        "summary": FeatureMatrix(
            np.concatenate(
                [aggregates, numeric_summary, occurrence_summary], axis=1
            ),
            aggregate_names(timed)
            + _summary_names(numeric_variables, numeric_statistics)
            + _summary_names(occurrence_variables, occurrence_statistics),
        )
    }
    for segmented in SEGMENTED_BY_PROFILE[temporal_profile]:
        # The segment boundary is a property of the sample, not of any one
        # code: the latest event time across every code.
        last_event_hours = np.zeros(rows, dtype=np.float64)
        for (sample, _code), series in groups.items():
            for row in series:
                if row[0] > last_event_hours[sample]:
                    last_event_hours[sample] = row[0]
        segment_matrix = build_segment_representation(
            groups,
            rows,
            numeric_variables,
            occurrence_variables,
            numeric_global,
            occurrence_global,
            sample_window_hours,
            last_event_hours,
            segmented,
            # The visit_halves boundary is fitted on the training split and
            # applied unchanged elsewhere, so an explicit boundary overrides
            # the cohort computation for validation/test featurization.
            (
                cohort_halves_split_hours(sample_window_hours)
                if halves_split_hours is None
                else float(halves_split_hours)
            ),
            timed,
        )
        built[segmented] = FeatureMatrix(
            np.concatenate([aggregates, segment_matrix.values], axis=1),
            aggregate_names(timed) + segment_matrix.names,
        )
    return built
