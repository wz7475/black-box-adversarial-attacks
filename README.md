# AMHE attack
<img width="911" height="279" alt="image" src="https://github.com/user-attachments/assets/382c7e3f-21dc-42cc-b6a8-2f8374a238ff" />

<img width="806" height="237" alt="image" src="https://github.com/user-attachments/assets/92d60d54-29a1-4a2c-994f-96370c59949a" />

<img width="770" height="431" alt="image" src="https://github.com/user-attachments/assets/c9105324-bb87-4901-8f01-2544b22d8b61" />


## setup
make sure to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) and run
```shell
uv sync
uv build
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
