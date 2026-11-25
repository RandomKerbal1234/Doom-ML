import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
METRICS_PATH = ROOT / "training" / "output" / "metrics.csv"


def save_config(config):
	with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
		json.dump(config, config_file, indent=2)
		config_file.write("\n")


def read_best_fitness():
	with METRICS_PATH.open("r", newline="", encoding="utf-8") as metrics_file:
		rows = csv.DictReader(metrics_file)
		last_row = None
		for row in rows:
			last_row = row

	if last_row is None or "best_fitness" not in last_row:
		raise ValueError(f"No best_fitness value found in {METRICS_PATH}")
	return float(last_row["best_fitness"])


def main():
	with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
		config = json.load(config_file)

	original_generations = config["generations"]
	config["generations"] = 1
	save_config(config)

	try:
		while True:
			subprocess.run(
				["python3", "training/train.py", "--fresh"],
				cwd=ROOT,
				check=True,
			)
			best_fitness = read_best_fitness()
			if best_fitness > 500:
				break

			print(f"Seed: {config['seed']}, best_fitness: {best_fitness}")
			config["seed"] += 10
			save_config(config)
	finally:
		config["generations"] = original_generations
		save_config(config)


if __name__ == "__main__":
	main()
