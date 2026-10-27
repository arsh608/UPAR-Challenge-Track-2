# UPAR 2027 Track 2 – Attribute-based Person Retrieval

**Method:** fine-tune OpenAI CLIP ViT-B/16 to predict the 40 UPAR attributes, then rank the gallery by
how likely each image is to match the query exactly.

| Component | Choice | Why |
|---|---|---|
| Backbone | CLIP ViT-B/16, 256x128 input (position grid resampled 14x14 → 16x8) | Web-scale pretraining generalises to the unseen test domain |
| Head | [CLS ; mean patch] → LayerNorm → Linear(40) | Global + part evidence |
| Training | BCE, AdamW, layer-wise lr decay 0.75, cosine, EMA weights, AMP | Standard stable CLIP fine-tuning |
| Augmentation | flip, scale/shift, **low-resolution simulation**, brightness/contrast, small erasing, *no hue jitter* | Surveillance crops are small and blurry; colour is a label |
| Retrieval score | log-likelihood of the query (+ α · expected agreement) + β · query-side log-softmax | The baseline's L1 distance ignores confidence; each image matches exactly one query |
| Tuning | α, β, τ, ε grid-searched on the validation split | Official val data, allowed by the rules |
| Inference | flip TTA, fp16, threaded image loading | Fits the 5-minute limit |

## Files
- `upar_model.py`: CLIP ViT in plain torch plus the attribute head (also shipped in the submission)
- `retrieval.py`: scoring and an exact re-implementation of mADM/mAP/Rank-k/mINP (checked against the formulas)
- `prepare_cache.py`: resizes every image once into a ~1 GB JPEG cache
- `data.py`: dataset and GPU augmentation
- `train.py`: training (resumable)
- `evaluate.py`: validation metrics for Track 1 and Track 2 plus the scoring grid search; writes `scoring.json`
- `analysis.py`: report figures, scoring ablation and the domain-shift robustness check (CPU only)
- `build_submission.py`: builds the Codabench zip (`run.py` + one or more models + scoring; the final submission is a 2-model ensemble)
- `submission_src/run.py`: the code Codabench runs
- `fetch_clip.py`: exports the CLIP visual weights
- `colab_train.ipynb`: runs the whole pipeline on Colab

## Run
Open `colab_train.ipynb` in Colab and run the cells top to bottom (see the notebook for the Drive layout).
