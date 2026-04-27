import mlflow
import hashlib
import os
import pandas as pd
from rich.progress import track
from typing import List, Tuple


def get_experiments_by_name(experiment_name: str):
    """
    Retrieves a list of experiments with the given name.

    Args:
        experiment_name (str): The name of the experiment to search for.

    Returns:
        List[mlflow.entities.Experiment]: A list of experiments with the given name.
    """
    return mlflow.search_experiments(filter_string=f"name='{experiment_name}'")


def get_run_ids(experiment_id: str):
    """
    Retrieves a list of run ids from mlflow for the given experiment id.

    Args:
        experiment_id (int): The id of the experiment to search for.

    Returns:
        List[str]: A list of run ids. If experiment_id is None, returns None.
    """
    if experiment_id is None:
        return None

    print(f"Searching experiment with id {experiment_id}")
    df = mlflow.search_runs([str(experiment_id)])

    print(f"Found {len(df)} runs")

    if len(df[(df.status != "FINISHED")]) > 0:
        print(f"{df[(df.status != 'FINISHED')]} runs not finished")
    else:
        print("All runs finished")

    return df.run_id.to_list()


def generate_hash(run_ids: List[str]):
    """
    Generates a hash of the given list of run ids.

    Args:
        run_ids (List[str]): List of run ids to generate the hash for.

    Returns:
        str: The hash of the given run ids.
    """
    hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
    return hs


def cache_df(run_ids: List[str], df=None, use_cache=True):
    """
    This function takes a list of run ids and an optional dataframe.
    It calculates the hash of the run ids and checks if a dataframe with the same hash already exists.
    If it does, it reads the dataframe from the cache and returns it.
    If it doesn't, it generates the dataframe using generate_df and saves it to the cache before returning it.

    The cache is stored as a pickle file because the dataframe contains
    list-valued columns (per-step metric histories) which would not survive
    a CSV round-trip.

    Args:
        run_ids (List[str]): A list of run ids to generate the dataframe from.
        df (pd.DataFrame, optional): An optional dataframe to use instead of generating a new one.

    Returns:
        pd.DataFrame: The generated dataframe.
    """

    # calculate hash
    hs = generate_hash(run_ids)

    # save df to cache
    path = f".cache/{hs}/"
    os.makedirs(path, exist_ok=True)

    cache_file = f"{path}df.pkl"

    if use_cache and os.path.exists(cache_file):
        print(f"DF already exists: {hs}")
        df = pd.read_pickle(cache_file)
    else:
        if df is None:
            df = generate_df(run_ids)
        df.to_pickle(cache_file)
        print(f"Created DF cache: {hs}")

    return df, hs


def generate_df(run_ids: List[str]):
    """
    This function takes a list of run ids and generates a dataframe from them.
    It checks if each run is finished and if not, it adds the run id to the list of unfinished runs.
    For each finished run, it extracts relevant parameters and metrics and adds them to the dataframe.
    The dataframe is then returned, along with the list of unfinished runs.

    Args:
        run_ids (List[str]): A list of run ids to generate the dataframe from.

    Returns:
        tuple: A tuple containing the generated dataframe and the list of unfinished runs.
    """
    client = mlflow.tracking.MlflowClient()

    # Per-step metric histories are folded into the main DataFrame as two
    # list-valued columns per metric: "<metric>.steps" and "<metric>.values".
    stepwise_metric_names = [
        "train_mse",
        "pulse_scaler_mean",
        "pulse_scaler_std",
        "rank.r_theta", 
        "rank.r_ext",   
        "rank.sv_theta",
        "rank.sv_ext",  
    ]

    rows = []
    unfinished_runs = []
    for run_id in track(run_ids, description="Collecting data..."):
        # obtain run from mlflow api
        run = client.get_run(run_id)

        # check if run is finished
        if run.info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            unfinished_runs.append(run_id)
            continue

        row = {
            "run_id": run_id,
            # get all relevant parameters
            "ansatz": run.data.params["model.circuit_type"],
            "model.seed": int(run.data.params["model.seed"]),
        }

        if "data.seed" in run.data.params:
            row["data.seed"] = int(run.data.params["data.seed"])
        if "model.n_pulse_params" in run.data.params:
            row["model.n_pulse_params"] = int(run.data.params["model.n_pulse_params"])
            row["model.n_gate_params"] = int(run.data.params["model.n_gate_params"]) # change to params

        frequencies = sorted(
            [
                float(f_key.replace("coeff.var.f", ""))
                for f_key in run.data.metrics.keys()
                if "coeff.var.f" in f_key
            ]
        )

        for f in frequencies:
            if f >= 0:
                row[f"coeff.var.f{f}"] = float(run.data.metrics[f"coeff.var.f{f}"])
                row[f"coeff.mean.f{f}"] = float(run.data.metrics[f"coeff.mean.f{f}"])

        # get metrics
        if "fcc" in run.data.metrics:
            row["fcc"] = float(run.data.metrics["fcc"])
            row["fcc.seed"] = int(run.data.params["fcc.seed"])
            row["pulse_params_variance"] = float(
                run.data.params["fcc.pulse_params_variance"]
            )

        if "expressibility" in run.data.metrics:
            row["expressibility"] = float(run.data.metrics["expressibility"])
            row["expressibility.seed"] = int(run.data.params["expressibility.seed"])
            row["pulse_params_variance"] = float(
                run.data.params["expressibility.pulse_params_variance"]
            )

        if "train_mse" in run.data.metrics:
            row["train_mse"] = run.data.metrics["train_mse"] # this will always return the last train_mse

        if "train.train_pulse" in run.data.params:
            if run.data.params["train.train_pulse"].lower() == "true":
                row["train_pulse"] = True
                row["pulse_scaler_mean"] = run.data.metrics["pulse_scaler_mean"]
                row["pulse_scaler_std"] = run.data.metrics["pulse_scaler_std"]
            else:
                row["train_pulse"] = False

        row["decompose_circuit"] = False
        if "model.decompose_circuit" in run.data.params:
            row["decompose_circuit"] = run.data.params["model.decompose_circuit"].lower() == "true"


        if "fidelity" in run.data.metrics:
            row["fidelity"] = float(run.data.metrics["fidelity"])
            row["fidelity.seed"] = int(run.data.params["fidelity.seed"])
            row["pulse_params_variance"] = float(
                run.data.params["fidelity.pulse_params_variance"]
            )

        if "trace-distance" in run.data.metrics:
            row["trace-distance"] = float(run.data.metrics["trace-distance"])

        # Fetch per-step metric histories and store as list-valued columns
        # "<metric>.steps" and "<metric>.values" so they live alongside the
        # summary metrics in the single DataFrame.
        for metric_name in stepwise_metric_names:
            history = client.get_metric_history(run_id, metric_name)
            if not history:
                print(f"No history for {metric_name} in {run_id}")
                continue
            # Sort by step to keep the arrays well-ordered
            history = sorted(history, key=lambda m: m.step)
            row[f"{metric_name}.steps"] = [int(m.step) for m in history]
            row[f"{metric_name}.values"] = [float(m.value) for m in history]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def export_csv(df: pd.DataFrame, name: str, experiment_id: str, hash: str):
    """
    Exports the given dataframe as a clean CSV file to the same folder
    in which figures are stored.

    Args:
        df (pd.DataFrame): The dataframe to export.
        name (str): The scenario name (e.g. 'study-1').
        experiment_id (str): The experiment id.
        hash (str): The hash of the run ids.
    """
    path = f"results/{experiment_id}/{hash}/"
    os.makedirs(path, exist_ok=True)
    filepath = f"{path}{name}.csv"
    df.to_csv(filepath, index=False)
    print(f"Exported CSV to {filepath}")
