# RGPO (Reliability-Guided Preference Optimization)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment

Create a `.env` file:
```bash
hf_key=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

## Step 3: Preprocess Datasets

Run `data/data_preprocess.ipynb` to standardize the raw datasets

## Step 4: Generate Reliability-based Preference Pairs

```bash
# HelpSteer2 - helpfulness (with ties)
python maximum_like_est/mle_model.py \
    --dataset helpsteer2 \
    --data_path data/train/helpsteer2_disagreement_paired.json \
    --annotation_dim helpfulness

# MultiPref - helpful (with ties)
python maximum_like_est/mle_model.py \
    --dataset multipref \
    --data_path data/train/multipref_combined.json \
    --annotation_dim helpful
```

## Step 5: Train Model

```bash
# Train on HelpSteer2
bash run_rgpo_helpsteer2.sh

# Train on MultiPref
bash run_rgpo_multipref.sh
```

## Step 6: Evaluate Models

```bash
# AlpacaEval 2.0
bash eval/run_alpaca_eval.sh

# Arena-Hard
# Please clone the official Arena-Hard-Auto repository first.
# Then run the evaluation script:
bash eval/run_arena_hard.sh
```
