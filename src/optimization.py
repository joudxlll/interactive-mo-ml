# %%
from __future__ import annotations
from itertools import combinations
import logging
import os
import time
import random
from OP import GeneticAlgorithm, SimulatedAnnealing

import numpy as np

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.model.random_model import RandomModel

from utils.argparse import parse_args
from utils.common import make_dir
from utils.dataset import load_dataset_from_openml
from utils.optimization import multi_objective, single_objective
from utils.pareto import (
    encode_pareto,
    get_pareto_from_history,
    plot_pareto_from_history,
    plot_pareto_from_smac,
    get_pareto_indicators,
)
from utils.preference_learning import get_preference_budgets
from utils.sample import grid_search, random_search
from utils.input import ConfDict, create_configuration
from utils.output import (
    adapt_paretos,
    check_pictures,
    save_paretos,
    check_dump,
    load_dump,
    update_config,
)


if __name__ == "__main__":
    args, _ = parse_args()
    create_configuration(
        file_name=args.conf_file,
        origin="optimization",
    )

    random.seed(ConfDict()["seed"])
    np.random.seed(ConfDict()["seed"])

    preference_budgets = get_preference_budgets()

    for main_indicator in get_pareto_indicators().keys():
        for mode in ["indicators", "preferences"]:
            for preference_budget in preference_budgets:
                for seed in [0, 1, 42]:
                    # Call your optimization methods here instead of single_objective
                    optimizer = GeneticAlgorithm(objective_function, population_size, mutation_rate, crossover_rate)
                    best_solution = optimizer.run(max_generations)
                    # Use the obtained best_solution in the remaining code if needed
                    # ...

    # for mode in ["fair", "unfair"]:
    #     for preference_budget in preference_budgets:
    #         multi_objective(mode=mode, preference_budget=preference_budget)


# %%
# Collect hyperparameters and fitness values from GA optimization
ga_hyperparameters = {...}  # Dictionary containing hyperparameters used in GA optimization
ga_fitness_value = ...  # Fitness value obtained from GA optimization

# Retrieve SMAC optimized hyperparameters
scenario = Scenario({"run_obj": "quality", "runcount-limit": 1})
hpo_facade = HPOFacade(scenario=scenario, rng=np.random.RandomState(0), tae_runner=RandomModel())
smac_hyperparameters = hpo_facade.optimize().get_dictionary()


# Calculate fitness value using SMAC optimized hyperparameters
smac_fitness_value = calculate_fitness(smac_hyperparameters)
ga_fitness_value = objective_function(best_solution)
# Compare fitness values
if ga_fitness_value > smac_fitness_value:
    print("GA optimization performed better with a fitness value of", ga_fitness_value)
else:
    print("SMAC optimization performed better with a fitness value of", smac_fitness_value)
