import glob
import os
import yaml
import shutil
import argparse
from rich.progress import track


def main(mlflow_path, backup_dir, dry_run, exclude_dirs, cut_after):

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    runs = glob.glob(os.path.join(mlflow_path, "*"))
    print(f"Found {len(runs)} runs")

    for r in track(runs, description="Cleaning up...", total=len(runs)):
        mark_for_deletion = False
        mark_for_deprecation = False
        if not os.path.isdir(r):
            continue
        if any([os.path.isdir(os.path.join(mlflow_path, x)) for x in exclude_dirs]):
            continue
        with open(os.path.join(r, "meta.yaml"), "r") as f:
            try:
                content = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

            if content["status"] != 3:
                mark_for_deletion = True
            elif int(content["end_time"]) < cut_after:
                mark_for_deprecation = True
                content["lifecycle_stage"] = "deleted"

        if mark_for_deletion:
            print(f"Moving {r} to trash")
            if not dry_run:
                shutil.move(r, os.path.join(backup_dir, os.path.basename(r)))
        elif mark_for_deprecation:
            print(f"Marking {r} as deprecated. Will be deleted next run.")
            if not dry_run:
                with open(os.path.join(r, "meta.yaml"), "w") as f:
                    yaml.safe_dump(content, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cut-after", type=int, default=0)
    parser.add_argument("--exclude-dirs", nargs="+", default=["tags"])
    parser.add_argument("--mlflow-path", type=str, default="./mlruns")
    parser.add_argument("--backup-dir", type=str, default="./mlruns_bckp")

    args = parser.parse_args()
    experiment_id = args.experiment_id

    if experiment_id is None:
        print("No experiment id provided. Use --experiment-id <experiment_id>")
        exit()

    mlflow_path = f"{args.mlflow_path}/{experiment_id}"
    backup_dir = f"{args.backup_dir}/{experiment_id}"
    dry_run = args.dry_run
    exclude_dirs = args.exclude_dirs
    cut_after = args.cut_after

    print("-" * 100)
    print(f"MLflow path: {mlflow_path}")
    print(f"Backup path: {backup_dir}")
    print(f"Dry run: {dry_run}")
    print(f"Exclude dirs: {exclude_dirs}")
    print(f"Cut after: {cut_after}")
    print("-" * 100)

    main(
        mlflow_path=mlflow_path,
        backup_dir=backup_dir,
        dry_run=dry_run,
        exclude_dirs=exclude_dirs,
        cut_after=cut_after,
    )
