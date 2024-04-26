import itertools
import json
import os
import sys
from pathlib import Path

def generate_permutations(parameters):
    keys = parameters.keys()
    values = parameters.values()
    for instance in itertools.product(*values):
        permutation = dict(zip(keys, instance))
        # Permute parameters under 'contrib_model_params' if present
        if 'contrib_model_params' in permutation:
            contrib_params = permutation['contrib_model_params']
            contrib_keys = contrib_params.keys()
            contrib_values = contrib_params.values()
            for contrib_instance in itertools.product(*contrib_values):
                contrib_permutation = dict(zip(contrib_keys, contrib_instance))
                yield {**permutation, 'contrib_model_params': contrib_permutation}
        else:
            yield permutation

def save_to_json(permutations, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    if len(os.listdir(directory)) > 0:
        raise Exception("Directory is not empty. Please specify an empty directory to avoid overwriting files.")

    for i, permutation in enumerate(permutations):
        with open(f"{directory}/job_{i+1}.json", "w") as file:
            json.dump(permutation, file, indent=4)

def main(directory):
    # Define the parameters dictionary with contrib_model_params as a list
    parameters = {
        "gauge_id": ["123"],
        "data_file": ["data/catchments/1.json"],
        "columns_file": ["data/columns_ecmwf.txt"],
        "aws_bucket": ["ecmwf-weather-data"],
        "epochs": [1],
        "batch_size": [1],
        "combiner_train_stride": [1],
        "combiner_holdout_size": [1],
        "contrib_model_variation": ["Transformer"],
        "use_future_covariates": [False],
        "contrib_model_params": [
            {
                "random_state": [1],
                "input_chunk_length": [1],
                "output_chunk_length": [1],
                "d_model": [1],
                "nhead": [1],
                "num_encoder_layers": [1],
                "num_decoder_layers": [1],
                "dim_feedforward": [1],
                "dropout": [1],
                "activation": ["relu"],
                "n_epochs": [1],
                "force_reset": [True],
                "pl_trainer_kwargs": [
                    {
                        "accelerator": ["gpu"],
                        "enable_progress_bar": [False]
                    }
                ]
            }
        ],
        "forecast_horizon": [1],
        "test_start": [1],
        "test_stride": [1]
    }

    # Generate all permutations of parameters
    permutations = generate_permutations(parameters)
    # Save each permutation to a JSON file in the specified directory
    save_to_json(permutations, directory)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_jobs.py path/to/dir/")
        sys.exit(1)

    main(sys.argv[1])