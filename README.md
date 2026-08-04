# Character-Level Symbolic Recurrence Biomarker for Precision Assessment of Cognitive Decline

Code for: "Character-level linguistic biomarkers for precision assessment of cognitive decline: a symbolic recurrence approach" (Frontiers in Aging Neuroscience, 2025).

## Overview

This repo implements a character-level symbolic recurrence representation for analyzing speech transcripts in the context of cognitive decline. The core idea: encode transcripts as one-hot character sequences, compute recurrence plots via epsilon-thresholded pairwise distances, and extract features from the resulting structure.

**Important note on encoding:** The recurrence computation uses one-hot character vectors, not raw integer character codes. With one-hot encoding, all non-matching character pairs have identical distance (sqrt(2)), making the epsilon threshold operationally equivalent to exact symbolic equality. This removes ordinal artifacts that would arise from alphabetical ordering of integer codes.

## Repo Structure

```
symbolic-recurrence-biomarker/
    src/
        pipeline.py          # Core functions: preprocessing, recurrence,
                             # Siamese network, evaluation utilities
    notebooks/
        corrected_cv.ipynb   # Stratified 5-fold CV with Siamese retrained
                             # per fold (corrected evaluation)
    data/
        README.md            # Instructions for obtaining the dataset
    requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The DementiaBank Pitt Corpus cannot be redistributed. Apply for access at https://dementia.talkbank.org/, then place your prepared CSV in `data/`. See `data/README.md` for the expected format.

## Usage

Run `notebooks/corrected_cv.ipynb` for the full evaluation pipeline.

Or use the components directly:

```python
from src.pipeline import (
    preprocess_text,
    build_vocab,
    characters_to_onehot,
    calculate_recurrence_matrix,
    recurrence_to_image,
    train_siamese_and_embed
)

# preprocess
characters = preprocess_text("the boy is falling off the stool")
char_to_index, vocab_size = build_vocab([characters])

# encode and compute recurrence
onehot = characters_to_onehot(characters, char_to_index, vocab_size)
recurrence = calculate_recurrence_matrix(onehot)
image = recurrence_to_image(recurrence, target_size=128)
```

## Evaluation Note

The original paper reported AUC from a pipeline where the Siamese network was trained on the full dataset before cross-validating the downstream XGBoost classifier. The corrected evaluation in this repo retrains the Siamese inside each fold on train-only data, which is the proper setup for unbiased performance estimation.

## Related Work

For a more mathematically grounded approach to character-level biomarker extraction, see our CharMark paper.

## Data

The dataset is derived from the DementiaBank Pitt Corpus (Cookie Theft task). Our sample includes broader ADRD diagnoses, not only ProbableAD/PossibleAD. The corpus cannot be redistributed; access requires DementiaBank membership: https://dementia.talkbank.org/

## Citation

```bibtex
@article{mekulu2025character,
  title={Character-level linguistic biomarkers for precision assessment of cognitive decline: a symbolic recurrence approach},
  author={Mekulu, Kevin},
  journal={Frontiers in Aging Neuroscience},
  volume={17},
  pages={1681124},
  year={2025}
}
```
