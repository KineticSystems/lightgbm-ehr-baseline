#!/usr/bin/env python3
"""Verify the upstream-attributable split=None event-encoding repair."""

from __future__ import annotations

import json
from types import SimpleNamespace

import src.trainer.base as base


class FixtureDataset:
    """Minimal dataset boundary used without opening clinical data."""

    constructions: list[object] = []

    def __init__(self, args, split, data, frame):
        self.args = args
        self.split = split
        self.data = data
        self.frame = frame
        self.collate_fn = object()
        self.__class__.constructions.append(self)


def main() -> None:
    trainer = base.Trainer.__new__(base.Trainer)
    trainer.args = SimpleNamespace(src_data="mimiciv", batch_size=8)
    trainer.dataset = FixtureDataset
    trainer.data = object()
    trainer.df = object()

    calls: list[tuple[FixtureDataset, dict[str, object]]] = []
    original_dataloader = base.DataLoader

    def fixture_dataloader(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return SimpleNamespace(dataset=dataset, kwargs=kwargs)

    try:
        base.DataLoader = fixture_dataloader

        # The released non-MEDS encoder deliberately asks for split=None.
        loader = trainer.dataloader_set(None)
        assert loader is not None
        assert len(FixtureDataset.constructions) == 1
        assert FixtureDataset.constructions[0].split is None
        assert calls[0][1]["batch_size"] == 8
        assert calls[0][1]["shuffle"] is False
        assert calls[0][1]["num_workers"] == 8

        # The later MEDS path is the sole path for which None means no split.
        trainer.args.src_data = "meds"
        assert trainer.dataloader_set(None) is None
        assert len(FixtureDataset.constructions) == 1
        assert len(calls) == 1
    finally:
        base.DataLoader = original_dataloader

    print(
        json.dumps(
            {
                "mimiciv_split_none_loader": "constructed",
                "meds_split_none_loader": None,
                "status": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
