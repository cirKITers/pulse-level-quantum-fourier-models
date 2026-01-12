import mlflow
import hashlib
import os
import pandas as pd
from rich.progress import track
from typing import List


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


def cache_df(run_ids: List[str], df=None):
    """
    This function takes a list of run ids and an optional dataframe.
    It calculates the hash of the run ids and checks if a dataframe with the same hash already exists.
    If it does, it reads the dataframe from the cache and returns it.
    If it doesn't, it generates the dataframe using generate_df and saves it to the cache before returning it.

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

    if os.path.exists(f"{path}df.csv"):
        print(f"DF already exists: {hs}")
        df = pd.read_csv(f"{path}df.csv")
    else:
        if df is None:
            return generate_df(run_ids), hs
        df.to_csv(f"{path}df.csv")
        print(f"Created DF cache: {hs}")
        df = pd.read_csv(f"{path}df.csv")

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
    df = pd.DataFrame()

    unfinished_runs = []
    for it, run_id in track(
        enumerate(run_ids), description="Collecting data...", total=len(run_ids)
    ):
        # obtain run from mlflow api
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        # check if run is finished
        if run.info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            unfinished_runs.append(run_id)
            continue

        # set run_id
        df.loc[it, "run_id"] = run_id

        # get all relevant paramters
        df.loc[it, "ansatz"] = run.data.params["model.circuit_type"]
        df.loc[it, "encoding_strategy"] = run.data.params["model.encoding_strategy"]
        df.loc[it, "data.seed"] = run.data.params["data.seed"]
        df.loc[it, "model.seed"] = run.data.params["model.seed"]
        df.loc[it, "fcc.seed"] = run.data.params["fcc.seed"]

        # get metrics
        df.loc[it, "fcc"] = run.data.metrics["fcc"]
        df.loc[it, "train_mse"] = run.data.metrics["train_mse"]

    return df
