# Set-up

Run the following to install the environment. Training downloads `compling/cs489-f26-a2` when it starts. The submitted configs log to Weights & Biases in offline mode. Export an API key as environment variable in the shell for online logging.

```bash
uv sync
export WANDB_API_KEY="<your-wandb-api-key>"
```

# Part 1: LSTM

Run the following to train the LSTM and save the best development checkpoint. It reads `artifacts/part1/config.yaml`. The checkpoint is saved to `artifacts/part1/checkpoint/model.pt`, and the metrics to `results/part1.csv`.

```bash
uv run python run_part1.py --config artifacts/part1/config.yaml
```

Run the following to load that checkpoint and print development loss and perplexity. It reads `artifacts/part1/config.yaml` and `artifacts/part1/checkpoint/model.pt`.

```bash
uv run python eval_checkpoint.py --config artifacts/part1/config.yaml \
    --checkpoint artifacts/part1/checkpoint/model.pt
```

# Part 2: Transformer

Run the following to train the Transformer and save the best development checkpoint. It reads `artifacts/part2/config.yaml`. The checkpoint is saved to `artifacts/part2/checkpoint/model.pt`, and the metrics to `results/part2.csv`.

```bash
uv run python run_part2.py --config artifacts/part2/config.yaml
```

Run the following to load that checkpoint and print development loss and perplexity. It reads `artifacts/part2/config.yaml` and `artifacts/part2/checkpoint/model.pt`.

```bash
uv run python eval_checkpoint.py --config artifacts/part2/config.yaml \
    --checkpoint artifacts/part2/checkpoint/model.pt
```
