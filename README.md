# Explainable Chest X-ray Retrieval RAG (PadChest)

An end-to-end Python pipeline for chest X-ray similarity retrieval and patient-friendly explanation generation.

This project:
- Loads PadChest-style metadata and image files
- Supports zip-wise embedding generation for very large datasets
- Generates DenseNet121 embeddings (PyTorch)
- Fits one global PCA model across all parts
- Indexes vectors with FAISS (L2)
- Retrieves top-k similar X-rays for a query image
- Resolves and displays retrieved top-k images from your local image root
- Builds structured RAG context (labels, locations, report snippets)
- Generates cautious, patient-friendly explanations (OpenAI optional)
- Provides both CLI and Streamlit UI

## Features

- DenseNet121 feature extraction with batch inference and GPU support
- PCA dimensionality reduction (default 256)
- FAISS IndexFlatL2 indexing and search
- Robust metadata parsing for list-like label/location fields
- Graceful handling of missing/corrupt images
- Streamlit UI with medical image display normalization for 16-bit grayscale images
- Validation and smoke-test utilities for artifact consistency

## Project Structure

- main.py: CLI entrypoint (`build`, `build-index`, `query`, `validate`)
- process_zip.py: Process one zip (`0.zip`, `1.zip`, etc.) into raw part embeddings
- pca_fit_transform.py: Fit one global PCA model and transform every part
- merge_artifacts.py: Merge all transformed part outputs into final artifacts
- artifact_utils.py: Shared helpers for part discovery, zip/part naming, and JSON utilities
- streamlit_app.py: Streamlit UI
- data_loader.py: CSV loading, path resolution, metadata parsing
- embedding.py: DenseNet121 embedding generation
- pca.py: PCA fit/transform/save/load
- faiss_index.py: FAISS index build/save/load
- retrieval.py: Query embedding + FAISS retrieval
- rag_explainer.py: RAG context + explanation generation
- smoke_test.py: Artifact validation and smoke query
- requirements.txt: Python dependencies
- .streamlit/config.toml: Streamlit runtime config

## Data Requirements

Your CSV should contain at least:
- ImageID
- Labels
- Localizations

Optional but supported:
- ImageDir
- Report

Image path resolution supports:
1. Absolute ImageID path
2. images_root/ImageID
3. images_root/ImageDir/ImageID

The loader prefers existing paths on disk and handles numeric ImageDir values safely.

## Installation

From the project folder:

```bash
python -m pip install -r requirements.txt
```

If you face dependency conflicts in a global environment, use a fresh conda/venv environment.

## CLI Usage

### 1) Zip-Wise Raw Embedding Generation

Run once per zip file.

```bash
python process_zip.py \
  --zip_path data/0.zip \
  --metadata_csv "C:/Users/Shiva/OneDrive/Pictures/Documents/Code/minor/chest_x_ray_images_labels_sample.csv" \
  --output_dir artifacts \
  --batch_size 32
```

Each run creates `artifacts/part_<zip_index>/` with:
- `embeddings.npy` (raw DenseNet embeddings)
- `metadata.json`
- `image_ids.npy`

### 2) Global PCA Fit + Transform (One Model For All Parts)

```bash
python pca_fit_transform.py \
  --output_dir artifacts \
  --n_components 256 \
  --mode incremental \
  --batch_size 2048
```

This saves `artifacts/pca_model.pkl` and `embeddings_pca.npy` inside every part directory.

### 3) Merge All Parts Into Final Retrieval Artifacts

```bash
python merge_artifacts.py --output_dir artifacts
```

This produces:
- `artifacts/final_embeddings.npy`
- `artifacts/final_metadata.json`

### 4) Build FAISS Index From Merged Embeddings

```bash
python main.py build-index --output-dir artifacts
```

### 5) Query Similar Cases

```bash
python main.py query \
  --query-image "C:/path/to/query.png" \
  --top-k 5 \
  --output-dir artifacts
```

Optional:
- --device cpu|cuda
- --disable-openai
- --openai-model gpt-4o-mini

### 6) Validate Artifacts

```bash
python main.py validate --output-dir artifacts
```

Optional smoke retrieval:

```bash
python main.py validate \
  --output-dir artifacts \
  --query-image "C:/path/to/query.png" \
  --top-k 3
```

## Streamlit UI

Run:

```bash
python -m streamlit run streamlit_app.py
```

UI supports:
- Build artifacts
- Validate artifacts
- Query retrieval
- Similar-case image thumbnails (resolved from local image root)
- Label-wise explanations
- Final patient-friendly explanation

Default paths in UI are prefilled for the sample dataset in this project.

### Local Retrieval Image Resolution

If retrieval metadata has empty `image_path` values (as expected for zip-wise builds), Streamlit resolves images using:
1. `Retrieved images root` + `train_folder` + `image_filename`
2. `Retrieved images root` + `image_filename`
3. `Retrieved images root` + `image_id`

`train_folder` and `image_filename` are created automatically by `process_zip.py`.
Use a local dataset layout like:
- `<images_root>/0/...images from 0.zip`
- `<images_root>/1/...images from 1.zip`
- `<images_root>/2/...`

## OpenAI (Optional)

To enable LLM-based explanation generation, set environment variable:

```bash
OPENAI_API_KEY=your_key_here
```

If not set, the app uses a local placeholder explainer with safe non-diagnostic language.

## Artifacts Output

By default, files are saved under artifacts/:
- part_0/embeddings.npy
- part_0/metadata.json
- part_0/image_ids.npy
- part_0/embeddings_pca.npy
- part_1/... (same structure)
- final_embeddings.npy
- final_metadata.json
- pca_model.pkl
- faiss.index

Raw per-part embeddings are preserved after PCA transform for reproducibility and optional future reprocessing.

## Troubleshooting

### No valid image embeddings were generated
- Verify dataset CSV path and image root path
- Check CSV column names in CLI/UI
- Ensure image files exist where resolver expects them

### Streamlit warning about torch.classes path
- Usually harmless
- Controlled by .streamlit/config.toml using poll file watcher

### Images look blank/washed out
- Medical PNGs can be 16-bit grayscale
- UI applies percentile windowing for display automatically

### pip dependency conflict warnings
- Common in base/global environments
- Prefer a clean virtual environment

## Safety Note

This system retrieves similar images and generates explanatory text. It does not provide medical diagnosis. Always confirm findings with a qualified clinician.
