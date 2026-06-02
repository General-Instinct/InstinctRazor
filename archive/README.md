# archive/ — quarantined dead-ends (kept for provenance, NOT on any live path)

Nothing here is used by the framework. Kept so the research history is auditable and so no one
re-discovers these the hard way. One line each on why it died.

| file | why it's here |
|------|---------------|
| `opd_train.py` | **device_map OPD trainer.** Naive 1-GPU pipeline (`device_map="auto"` puts only 1 of 4 GPUs active); OOMs/hangs at the first backward in the fused-expert forward (`opd_C.log` dies right after optimizer init, no step-0). **Superseded by `src/distill/opd_train_fsdp.py`** (FSDP2, 4-GPU, ~4× faster, even weight memory). |
| `chain_opd.sh` | driver for the dead `opd_train.py` (full A→B→C). Use `pipelines/distill.sh` instead. |
| `chain_opd_bc.sh` | device_map `opd_train.py` B→C rerun variant. Superseded by the FSDP path. |
| `chain_opd_c.sh` | device_map `opd_train.py` C-only rerun variant. Superseded by the FSDP path. |
| `smoke_fsdp.py` | one-off lossless-vs-device_map FSDP smoke (full-shard, same batch). Validation scratch, not in any chain — superseded by `opd_train_fsdp.py --smoke 2` (now exposed as `pipelines/distill.sh --smoke`). |
| `smoke_ep.py` | abandoned **expert-parallel (EP)** lossless smoke — an alternative to FSDP for the 4-GPU lossless forward. EP path not pursued (FSDP2 won on least-rewrite of the custom STE per-expert forward). |
| `lcb_diag.py` | LiveCodeBench extraction / `<think>` gate diagnostic. One-off; also crashed under vLLM spawn (missing `__main__` guard at the time). The LCB verdict (recoverable code-gap) is robust without it. |
| `diag_trunc.py` | one-off BBEH truncation/loop diagnostic. Its conclusion (genuine token-inefficiency under quantization) is already captured in `docs/OPD_INTEGRATION.md`. |

See also `src/distill/fsdp_setup.py`: an **orphan reference module** (imported by nobody) that
documents the verified FSDP2 unshard-before-forward mechanic. It is kept in `src/distill/` (not here)
because it is useful documentation, but the **live** wrapping logic lives inline in
`opd_train_fsdp.py:wrap_fsdp_opd()`.
