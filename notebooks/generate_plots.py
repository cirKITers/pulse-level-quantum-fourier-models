from data_helper import (
    get_experiments_by_name,
    get_run_ids,
    cache_df,
    export_csv,
)
from viz_helper import (
    save_figures,
    viz_study_1,
    viz_study_2,
    viz_study_3,
    viz_study_4,
)

# enable caching?
cache = False

# scenarios for plotting
scenarios = {
    "study-1": {
        "experiment_name": "study-1-14",
        "show_error": False,
        "threshold": 1e-4,
    },
    # "study-2": {
    #     "experiment_name": "study-2-2",
    #     "show_error": False,
    # },
    # "study-3": {
    #     "experiment_name": "study-3-0",
    #     "show_error": True,
    # },
    # "study-4": {
    #     "experiment_name": "study-4-4",
    #     "show_error": True,
    #     # "mse_step": 500,  # training step at which to evaluate MSE (None = final)
    # },
}

for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    ignore_ansatzes = setting.get("ignore_ansatzes", [])
    max_distortion = setting.get("max_distortion", 1.0)

    if not "experiment-id" in setting:
        print(
            f"No experiment id specified, searching for experiment with name: {setting['experiment_name']}"
        )
        setting["experiment_id"] = get_experiments_by_name(
            experiment_name=setting.get("experiment_name", "Default")
        )[0].experiment_id

    # Obtain all run ids for the specified experiment
    run_ids = get_run_ids(setting["experiment_id"])

    # Get the df from cache or generate it if it doesn't exist
    cache_id = setting
    df, hs = cache_df(run_ids, df=None)

    print(f"Hash for scenario: {hs}")

    print(f"Ignoring Ansatzes: {ignore_ansatzes}")
    for ansatz in ignore_ansatzes:
        df = df[df["ansatz"] != ansatz]

    if scenario == "study-1":
        figures = viz_study_1(
            df,
            max_distortion=max_distortion,
            threshold=setting["threshold"],
            show_error=setting["show_error"],
        )

    elif scenario == "study-2":
        figures = viz_study_2(
            df,
            max_distortion=max_distortion,
            show_error=setting["show_error"],
        )
    elif scenario == "study-3":
        figures = viz_study_3(
            df,
            max_distortion=max_distortion,
            show_error=setting["show_error"],
        )
    elif scenario == "study-4":
        figures = viz_study_4(
            df,
            show_error=setting["show_error"],
            mse_step=setting.get("mse_step", None),
        )

    save_figures(
        figures=figures,
        name=scenario,
        experiment_id=setting["experiment_id"],
        hash=hs,
    )

    export_csv(
        df=df,
        name=scenario,
        experiment_id=setting["experiment_id"],
        hash=hs,
    )
