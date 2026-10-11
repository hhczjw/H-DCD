# Research-one source and checkpoint publication

The current `coupled_BI_Mamba3` source is published without local reference literature, new dataset files, training caches or new result directories. Existing repository history is preserved.

The modified Mamba source is vendored in place of the previous gitlink, which had no matching `.gitmodules` entry. Its license is retained. This makes the local runtime modifications available to a fresh clone.

The historical Sept18 p0B seed2024 validation-Acc2-selected checkpoint is stored under `published_checkpoints/` using Git LFS. This is the checkpoint associated with the historical best Acc2 reference, not a claim that every metric is independently best. Current source/environment is not claimed to be a complete dated Sept18 snapshot.

Run `git lfs install` and `git lfs pull`. Dataset and pretrained paths need local configuration.
