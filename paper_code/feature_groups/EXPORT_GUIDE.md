# Exporting feature_groups to MEDS

There is nothing to run here. This paper's outcomes are evaluated on
the export in `../stepwise_embeddings`, which builds the same cohort.

## Where each outcome comes from

- `decompensation` -- use `stepwise_embeddings`'s `decompensation`
- `mortality` -- use `stepwise_embeddings`'s `ihm`

## Why

Both papers score the same MIMIC-III benchmark tasks. We compared the
two exports directly: every subject in `stepwise_embeddings`'s cohort also appears
in this paper's, under the same train/validation/test assignment, with
no subject assigned differently. `stepwise_embeddings`'s cohort is the smaller of
the two only because it is drawn down by the landmark sampling
contract described in its guide.

So the campaign fits that cohort once and reports the result for both
papers. Shipping a second exporter here would let you build a cohort
that no published number was fitted on, which is worse than shipping
nothing.

Follow `../stepwise_embeddings/EXPORT_GUIDE.md`. The baseline command there names
the task; the mapping above says which one answers for this paper.
