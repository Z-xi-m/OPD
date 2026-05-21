# THUNLP OPD Reproduction Backup

This note records the portable backup for continuing the THUNLP OPD reproduction on another machine.

## Backup Location

Main backup directory:

```text
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env
```

Environment package:

```text
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/envs/opd_verl_py312.tar
```

Environment metadata:

```text
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/manifests/opd_verl_py312_pip_freeze.txt
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/manifests/opd_verl_py312_versions.txt
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/manifests/thunlp_script_index.txt
/data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/manifests/thunlp_output_dir_index_depth2.txt
```

## Restore

Restore the environment to the same path if possible:

```bash
mkdir -p /dev/shm/zhu.ximo/conda_envs
tar -xf /data/zhu.ximo/opd_repro/backups/thunlp_opd/20260521_env/envs/opd_verl_py312.tar -C /dev/shm/zhu.ximo/conda_envs
source /dev/shm/zhu.ximo/conda_envs/opd_verl_py312/bin/activate
```

The backed up environment was:

```text
Python 3.12.13
torch 2.8.0+cu128
transformers 4.57.1
vllm 0.11.0
verl 0.7.0.dev
flash_attn 2.8.1
ray 2.55.1
```

## Important Paths

Remote source and scripts:

```text
/data/zhu.ximo/opd_repro/thunlp_opd
/data/zhu.ximo/opd_repro/thunlp_opd/scripts/repro/run_opd_paper_strict_8gpu_wandb.sh
/data/zhu.ximo/opd_repro/thunlp_opd/scripts/val/eval/gen_vllm.py
/data/zhu.ximo/opd_repro/thunlp_opd/scripts/val/eval/grade.py
```

The large THUNLP output tree was summarized into `docs/reproduction/` and then cleaned from `/data` to avoid keeping about 1.9T of intermediate checkpoints. Historical output paths before cleanup were:

```text
/data/zhu.ximo/opd_repro/outputs/thunlp_opd/checkpoints
/data/zhu.ximo/opd_repro/outputs/thunlp_opd/merged_hf
/data/zhu.ximo/opd_repro/outputs/thunlp_opd/eval_results
/data/zhu.ximo/opd_repro/outputs/thunlp_opd/validation_log
```

The lightweight retained record is:

```text
docs/reproduction/thunlp_opd_reproduction_record.md
docs/reproduction/thunlp_opd_fig2_repro_results.csv
docs/reproduction/thunlp_opd_weighted_summary.csv
docs/reproduction/thunlp_opd_fig2_weighted.svg
```

Model files are expected under:

```text
/data/zhu.ximo/model
```

## Notes For Continuing

- This is the first OPD reproduction project, based on `verl`.
- Do not treat `/data/zhu.ximo/opd_repro/g_opd` as this project's environment; the actual shared verl environment is `opd_verl_py312`.
- Keep W&B enabled for long training runs, but keep local log/cache directories on `/dev/shm` where possible.
- For result inspection, start from `CODE_READING_GUIDE.md`, then inspect the scripts under `scripts/repro` and `scripts/val`.
