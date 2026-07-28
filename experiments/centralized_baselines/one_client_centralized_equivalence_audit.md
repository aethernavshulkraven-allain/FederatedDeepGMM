# One-Client Centralized Equivalence Audit

Question: can `client_num_in_total=1` and `client_num_per_round=1` faithfully implement the centralized low-dimensional DeepGMM baselines by config only?

Short answer: it is a useful degenerate federated proxy for some centralized GDA/SGDA behavior, but it is not a fully faithful or paper-safe centralized baseline as-is. It cannot implement OAdam by config.

## Required One-Client Config Shape

The only config-only setting that gets close is:

```text
training_type = simulation
backend = sp
federated_optimizer = FedAvg
client_num_in_total = 1
client_num_per_round = 1
server_learning_rate = 1.0
```

For deterministic/full-batch behavior:

```text
batch_size = 0
client_optimizer = sgd
```

For stochastic/minibatch behavior:

```text
batch_size > 0
client_optimizer = sgd
```

This still runs through the federated `FedAvgAPI` path, not through a centralized runner.

## Data-Flow Equivalence

For low-dimensional `linear`, `abs`, `sin`, and `step`, `load_synthetic_data` calls `load_partition_data_mnist` and returns global train/dev/test splits plus local client dictionaries (`fedml/data/data_loader.py:369-421`).

Inside `fedml/data/MNIST/data_loader.py`, local data are distributed over `range(args.client_num_in_total)` (`MNIST/data_loader.py:17-20`). With exactly one client:


- the Dirichlet vector has length 1, so its only mass is 1;
- `train_samples_per_client[-1]` becomes the full train size;
- the same happens for test and dev;
- client 0 receives all samples, though permuted (`MNIST/data_loader.py:45-95`).

For `batch_size <= 0`, `load_synthetic_data` temporarily uses `batch_size=128` to construct loaders, then combines each local loader into a single tuple and restores the original batch size (`fedml/data/data_loader.py:375-381`, `fedml/data/data_loader.py:823-856`). Therefore, for one-client deterministic low-dimensional runs, client 0 sees all training samples in one full local batch.

Data verdict: one-client/full-batch is data-pooled enough for low-dimensional GDA-style smoke testing. `partition_alpha` becomes effectively irrelevant for the local split contents because all samples go to the only client, although permutations still occur.

## Training-Loop Flow

The active path is:

```text
main.py
  -> FedMLRunner
  -> SimulatorSingleProcess
  -> FedAvgAPI
  -> _setup_clients(...)
  -> per-round client.train(...)
  -> _aggregate(...)
  -> server delta update
```

`FedAvgAPI._setup_clients` creates client objects from local client dictionaries (`fedavg_api.py:342-363`). Each round samples clients, updates the local dataset, trains locally, aggregates local weights, and applies a server learning-rate update (`fedavg_api.py:399-493`).

With one client, `_client_sampling` returns `[0]` because `client_num_in_total == client_num_per_round` (`experiment_utils.py:175-180`). `_aggregate` averages a single local model, so the aggregate equals the local model (`fedavg_api.py:784-805`).

However, the server update still applies:

```text
theta_new = theta_old + server_learning_rate * (local_theta_after_train - theta_old)
```

for non-OGDA methods (`fedavg_api.py:471-488`). Therefore `server_learning_rate=1.0` is required for the post-aggregation global model to equal the one local model after centralized-looking training. Existing federated low-dimensional runs often use `server_learning_rate=1.5`, which is not directly equivalent to centralized optimizer updates.

Training-loop verdict: one-client with `server_learning_rate=1.0` removes multi-client aggregation, but it still travels through federated control flow and should be labeled carefully.

## Optimizer and Method Mapping

### `gda_d`

Closest config-only proxy:

```text
client_optimizer = sgd
batch_size = 0
server_learning_rate = 1.0
client_num_in_total = 1
client_num_per_round = 1
```

The GMM trainer uses `CustomSGD` for both `g` and `f` when `client_optimizer != "ogda"` (`fedavg_api.py:219-235`, `fedavg_api.py:244-251`). `train_gmm` loops through local data and applies optimizer steps to `g` and `f` (`my_model_trainer_classification.py:362-414`). The objective returns `(moment, -moment + f_reg)`, so minimizing the second objective corresponds to ascent on the moment with regularization (`simple_moment_objective.py:97-111`).

Faithfulness: plausible as a centralized GDA proxy only under strict settings, especially `server_learning_rate=1.0` and carefully matched iteration budget. It is not automatically the paper baseline because the runner metadata still calls it `fedgda_d` and the code executes the FedAvgAPI round structure.

### `sgda_s`

Closest config-only proxy:

```text
client_optimizer = sgd
batch_size > 0
server_learning_rate = 1.0
client_num_in_total = 1
client_num_per_round = 1
```

Client 0 receives all samples and `train_gmm` iterates through minibatches. This is close to centralized minibatch SGDA. Pitfall: for zoo datasets the DataLoader is converted to a list once (`fedml/data/data_loader.py:411-421`), so shuffling happens when the list is materialized, not freshly at every epoch/round. That is less faithful than a conventional centralized stochastic trainer with reshuffling per epoch.

Faithfulness: usable as an exploratory SGDA proxy, but not a clean paper-quality centralized SGDA implementation without documenting the fixed minibatch order and matching the training budget.

### `oadam_s` / `oadam_d`

Not implementable by config in the active low-dimensional path.

`OAdam` exists (`optimizers/oadam.py`), but `FedAvgAPI` only chooses `OGDA` when `client_optimizer == "ogda"`; otherwise it chooses `CustomSGD` for the GMM `g`/`f` updates (`fedavg_api.py:219-235`, `fedavg_api.py:244-251`). Setting `client_optimizer=oadam` would fall into the `else` branch and still use `CustomSGD` for the GMM game. `get_effective_config` also maps anything other than `ogda` to `fedgda`, not OAdam (`experiment_utils.py:85-95`).

Faithfulness: no. A true OAdam centralized baseline needs implementation changes.

### `ogda` / FedOGDA-style one-client

`client_optimizer=ogda` with one client gives a centralized-ish OGDA/FedOGDA proxy, but this is not the paper centralized OAdam baseline. Also, optimizer state is cleared in `train_gmm` at the start of each local train call (`my_model_trainer_classification.py:367-370`), while `FedAvgAPI` separately stores previous server deltas for OGDA-style server updates (`fedavg_api.py:446-470`). That is a specific FedOGDA-style control flow, not a clean direct centralized OGDA runner.

## Artifact and Metadata Pitfalls

One-client config-only runs would still produce effective configs like:

```text
algorithm = fedgda or fedogda
variant = fedgda_d / fedgda_s / fedogda_d / fedogda_s
federated_optimizer = FedAvg
client_num_in_total = 1
client_num_per_round = 1
```

This is not the centralized method naming needed for final paper tables (`gda_d`, `sgda_s`, `oadam_s`). Reporting these as centralized without either metadata fixes or a separate centralized runner would be confusing and risky.

The current `scripts/run_manifest.py` also filters to `training_scope=federated` and supported federated method names only, so existing centralized manifest rows cannot be launched by this path as-is (`scripts/run_manifest.py:116-130`).

## Main Pitfalls

1. `client_num_in_total=0` is invalid, not centralized.
2. `client_num_in_total=1` gives one local client, but the runner is still `FedAvgAPI`.
3. `server_learning_rate` must be `1.0`; existing values like `1.5` are not direct centralized updates.
4. OAdam cannot be selected by config.
5. Stochastic one-client runs use a minibatch list materialized once for zoo datasets, so minibatch reshuffling is not standard.
6. Metadata will label the run as federated unless changed.
7. The paper baselines mention extensive LR search and centralized DeepGMM/OAdam; a one-client FedAvg proxy should not be silently substituted.

## Verdict

`client_num_in_total=1` is not a fully faithful implementation of the centralized baselines required for the paper. It can be used as a clearly labeled one-client/full-batch proxy for centralized GDA/SGDA smoke testing under strict settings, but it should not be treated as the final centralized baseline implementation, and it cannot cover OAdam.

Recommended next step: if time is tight, run a tiny `one_client_proxy` smoke for `gda_d`/`sgda_s` only to sanity-check expected behavior, but implement or add a thin true centralized runner before reporting centralized baselines in final tables.
