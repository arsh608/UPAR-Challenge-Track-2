# UPAR Challenge 2027, Track 2: Task and Approach

**Team:** Arsh Al Aman (23K-0078) · Rao Ghulam Mohiuddin (23K-0001) · Sheikh Naveed Azeemi (23K-0003)  
**Course:** Computer Vision, FAST National University of Computer and Emerging Sciences · Instructor: Mr. Syed Bilal Ahsan  
**Codabench team:** `arshalaman` · [Track 2 leaderboard](https://www.codabench.org/competitions/18196/#/results-tab)

## Selected task
We chose Track 2 of the UPAR Challenge at the Real-World Surveillance workshop (RWS, WACV 2027): **attribute-based person retrieval**. A query is a description made of 40 binary attributes, such as age, gender, hair length, clothing type and colour, and accessories. For every query the system ranks a gallery of pedestrian crops so that the matching people come first. The gallery comes from an unseen surveillance domain, and the ranking metric is mADM, which gives partial credit for near misses. Solutions are submitted as code and run offline on the organisers' GPU within 5 minutes.

## Approach
1. **Attribute recogniser.** We fine-tune the visual encoder of OpenAI CLIP ViT-B/16 to predict the 40 attributes on 256×128 crops. Training uses BCE loss, layer-wise learning-rate decay and EMA weights. The augmentations suit surveillance footage: a low-resolution simulation for distant cameras and no hue changes, because colour is a label.
2. **Likelihood ranking.** We replace the starter kit's L1 distance with the log-likelihood that an image matches the whole query. We add an agreement term that rewards near misses and a normalisation across queries, because each image matches exactly one query. These settings were tuned on the validation split.
3. **Model selection.** Validation mADM peaks after 3 epochs and then declines, so all submitted models train for 3 epochs. The final submission averages a train-only model and a train+val model.

## Results
| | mADM | mAP | Rank-1 |
|---|---|---|---|
| Validation, organiser baseline | 0.113 | 0.001 | 0.000 |
| Validation, ours | 0.746 | 0.340 | 0.351 |
| **Hidden test, final ensemble (rank 32)** | **0.624** | **0.353** | **0.385** |
| Hidden test, winner | 0.699 | 0.416 | 0.397 |

## Deliverables
| Assignment requirement | Where |
|---|---|
| Brief description of task and approach | this file |
| Working code | `UPAR-Challenge-2027-main/solution/` (+ Colab notebook `colab_train.ipynb`) |
| Experimental results | `runs/` (training logs, validation results, ablation) and `report/figures/` |
| Report | `report/UPAR_Track2_Report.pdf` (7 pages; editable `.docx` next to it) |
| Proof of submission | `report/proof_of_submission/` + the Codabench results tab linked above |
| Demo presentation | `report/UPAR_Track2_Demo.pptx` |
