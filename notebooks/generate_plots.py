from data_helper import get_experiments_by_name, get_run_ids, cache_df
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
        "experiment_name": "study-1-10",
        "max_distortion": 0.01,
        "show_error": False,
        # "ignore_ansatzes": ["Circuit_10", "Circuit_3", "Circuit_18", "Circuit_7", "Circuit_9", "Circuit_16"]
        # "ignore_ansatzes": ["Circuit_13", "Circuit_17", "Circuit_8", "Hardware_Efficient"]
    },
    # "study-2": {
    #     "experiment_name": "study-2-3",
    #     "max_distortion": 0.01,
    #     "show_error": True,
    # },
    "study-3": {
        "experiment_name": "study-3-1",
        "max_distortion": 0.01,
        "show_error": True,
    },
}

for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    ignore_ansatzes = setting.get("ignore_ansatzes", [])

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
            max_distortion=setting["max_distortion"],
            show_error=setting["show_error"],
        )

    elif scenario == "study-2":
        figures = viz_study_2(
            df,
            max_distortion=setting["max_distortion"],
            show_error=setting["show_error"],
        )
    elif scenario == "study-3":
        figures = viz_study_3(
            df,
            max_distortion=setting["max_distortion"],
            show_error=setting["show_error"],
        )
    elif scenario == "study-4":
        figures = viz_study_4(
            df,
            show_error=setting["show_error"],
        )

    save_figures(
        figures=figures,
        name=scenario,
        experiment_id=setting["experiment_id"],
        hash=hs,
    )
