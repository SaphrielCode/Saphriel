SmolLM2-135M training assets

Local files (large assets are excluded from Git):
  data/checkpoints/step-240000 through step-1920000/
    All eight published intermediate checkpoint snapshots, including safetensors weights,
    architecture configuration and tokenizer. Step interval: 240,000.
    These are pretrained base-model checkpoints, not intermediate instruction-tuning stages.
    They do not include optimizer, scheduler or original random-number-generator state.
    No original step-zero initialization is included.

  data/datasets/smol-smoltalk/
    Complete published repository snapshot: supervised instruction-tuning data and test split.
  data/datasets/ultrafeedback_binarized/
    Complete published repository snapshot: preference-training data and associated splits.
    The official small-model DPO recipe uses train_prefs and test_prefs.

  data/pretraining-samples/
    FineWeb-Edu and DCLM: 1,000 complete JSONL records each.
    These are prefixes of the first public source shards, not random/representative samples.
    The samples are not proven to be the exact filtered documents used to train SmolLM2.
    The Stack was not downloaded: Hugging Face access approval is required.

The model card names FineWeb-Edu, DCLM and The Stack plus additional curated data.
These downloads do not reconstruct the complete original training mixture, exact filtered
subsets, sampling order, tokenization history or all unpublished/changed data.
Pinned current dataset snapshots need not match the precise historical revisions used in training.
The multi-terabyte pretraining repositories do not fit on this laptop.
Source sizes/access information: pretraining-sources.json.
Source sample provenance, hashes and selection limits: sample-receipt.json.
Checkpoint/dataset revisions and published verification hashes: manifest.json.
Completed-download verification receipt: download-receipt.json.
Training configurations and source URLs: reference/.
Reference recipes are published upstream examples, not ready-to-run Windows training launchers.

Restore downloads:
  python tools/download_training_assets.py
Already verified files are reused. Each newly downloaded file is checksum-verified.

Prepare the isolated readers on another checkout:
  python -m venv .venv-data
  .venv-data\Scripts\python.exe -m pip install -r requirements-data.txt
Capture bounded samples:
  .venv-data\Scripts\python.exe tools/sample_pretraining.py
Inspect checkpoint headers, Parquet metadata and sample integrity:
  .venv-data\Scripts\python.exe tools/check_training_assets.py

Original dataset cards in each local snapshot describe source licenses and restrictions.
Weights/datasets stay local. Git tracks download tools, pinned manifests, recipes and receipts.
No training or model modification has been performed by these downloads.
