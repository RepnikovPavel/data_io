# Scripts

## Download

```bash
nohup ./scripts/download_data.sh /mnt/hdd2/datasets_text > /tmp/hrm_text_download_data.log 2>&1 &
```

Один датасет отдельно (отладка/перезапуск): `./scripts/download_hf.sh REPO_ID [DATASETS_DIR] [LOCAL_SUBDIR]`, специальные: `download_amps.sh`, `download_scibench.sh`, `download_math_dataset.sh`. Лог каждого: `/tmp/hrm_text_download_<name>.log`.

## Train tokenizer

```bash
nohup ./scripts/train_tokenizer.sh /mnt/hdd2/datasets_text_transformed/HRM-Text /mnt/hdd2/models/HRM-Text/tokenizers/original/bpe > /tmp/hrm_text_train_tokenizer.log 2>&1 &
```

## Clean (всё по очереди)

```bash
nohup ./scripts/run_clean_all.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text > /tmp/hrm_text_clean_queue.log 2>&1 &
```

## Clean (по одному)

```bash
nohup ./scripts/clean_flan.sh /mnt/hdd2/datasets_text/Open-Orca/FLAN /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/flan > /tmp/hrm_text_clean_flan.log 2>&1 &
nohup ./scripts/clean_synth.sh /mnt/hdd2/datasets_text/PleIAs/SYNTH /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/SYNTH > /tmp/hrm_text_clean_synth.log 2>&1 &
nohup ./scripts/clean_acereason.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/acereason > /tmp/hrm_text_clean_acereason.log 2>&1 &
nohup ./scripts/clean_ampsmathematica.sh /mnt/hdd2/datasets_text/amps.tar.gz /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/ampsmathematica > /tmp/hrm_text_clean_ampsmathematica.log 2>&1 &
nohup ./scripts/clean_dmmath.sh /mnt/hdd2/datasets_text/mathematics_dataset-v1.0 /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/dmmath > /tmp/hrm_text_clean_dmmath.log 2>&1 &
nohup ./scripts/clean_openmathinstruct2.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/openmathinstruct2 > /tmp/hrm_text_clean_openmathinstruct2.log 2>&1 &
nohup ./scripts/clean_openthoughts2.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/openthoughts2 > /tmp/hrm_text_clean_openthoughts2.log 2>&1 &
nohup ./scripts/clean_sudoku.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/sudoku_extreme > /tmp/hrm_text_clean_sudoku.log 2>&1 &
nohup ./scripts/clean_tasksource.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/tasksource > /tmp/hrm_text_clean_tasksource.log 2>&1 &
nohup ./scripts/clean_textbookreasoning.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/textbookreasoning > /tmp/hrm_text_clean_textbookreasoning.log 2>&1 &
nohup ./scripts/clean_amps_khan.sh /mnt/hdd2/datasets_text/amps/khan /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_amps_khan.log 2>&1 &
nohup ./scripts/clean_gsm8k_train.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_gsm8k_train.log 2>&1 &
nohup ./scripts/clean_math_train.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_math_train.log 2>&1 &
nohup ./scripts/clean_natural_reasoning.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_natural_reasoning.log 2>&1 &
nohup ./scripts/clean_no_robots.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_no_robots.log 2>&1 &
nohup ./scripts/clean_numinamath.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_numinamath.log 2>&1 &
nohup ./scripts/clean_omnimath.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_omnimath.log 2>&1 &
nohup ./scripts/clean_principia_collection.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_principia_collection.log 2>&1 &
nohup ./scripts/clean_webinstruct_verified.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data > /tmp/hrm_text_clean_webinstruct_verified.log 2>&1 &
nohup ./scripts/clean_arb.sh /mnt/hdd2/datasets_text/Platypus/ARB /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_arb.log 2>&1 &
nohup ./scripts/clean_openbookqa.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_openbookqa.log 2>&1 &
nohup ./scripts/clean_reclor.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_reclor.log 2>&1 &
nohup ./scripts/clean_scibench.sh /mnt/hdd2/datasets_text/Platypus/scibench/dataset/original /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_scibench.log 2>&1 &
nohup ./scripts/clean_scienceqa.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_scienceqa.log 2>&1 &
nohup ./scripts/clean_theoremqa.sh /mnt/hdd2/datasets_text /mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus > /tmp/hrm_text_clean_theoremqa.log 2>&1 &
```
