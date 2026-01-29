from data_helper import get_run_ids, cache_df
from viz_helper import save_figures, coeff_var_over_distortion, fcc_over_distortion

# enable caching?
cache = False

# scenarios for plotting
scenarios = {"study-1": {"experiment_id": 896759427718134482, "max_distortion": 0.02}}

for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    # Obtain all run ids for the specified experiment
    run_ids = get_run_ids(setting["experiment_id"])

    # Get the df from cache or generate it if it doesn't exist
    cache_id = setting
    df, hs = cache_df(run_ids, df=None)

    print(f"Hash for scenario: {hs}")

    figures = []

    figures.append(fcc_over_distortion(df, max_distortion=setting["max_distortion"]))
    figures.append(
        coeff_var_over_distortion(df, max_distortion=setting["max_distortion"])
    )

    save_figures(
        figures=figures, name=scenario, experiment_id=setting["experiment_id"], hash=hs
    )
