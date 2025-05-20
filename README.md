# AMHE attack
## setup
make sure to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) and run
```shell
uv sync
```

to run scripts
```shell
source .venv/bin/activate
python <script.py> <args>
```
or add prefix `uv run`
```shell
uv run python <script.py> <args>
```

## usage
1. train models
```shell
python train_models.py
```
2. run attack (check help for detaul)
```shell
python main.py --help
python main.py --model mnist
```