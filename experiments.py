from runexp import Runner
import pandas as pd
import os
import json
import time

from models import AllocationModel, SchedulingModel
from utils import read_instance

"""
model = AllocationModel(tasks, calendars, same_allocation)
model.minimize(model.get_nb_teams_objective())
"""
class SolverRunner(Runner):

    def make_kwargs(self, config):

        tasks, calendars, same_allocation = read_instance(config['instance'])

        ModelClass: AllocationModel|SchedulingModel = eval(config['model_class'])
        model_kwargs = config['model_kwargs']
        model = ModelClass(tasks, calendars, same_allocation, **model_kwargs)

        objective = config['objective']
        if objective == "nb_teams":
            objective = model.get_nb_teams_objective()
        elif objective == "dispersion":
            objective = model.get_dispersion_objective()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        model.minimize(objective)

        return {
            "model": model,
            **config['solver_kwargs']
        }


def solve_model(model, solver, **solver_kwargs):

    t0 = time.time()
    res = model.solve(solver=solver, **solver_kwargs)
    t1 = time.time()
    return {
        "wallclock_time": t1 - t0,
        "solver_time": model.status().runtime,
        "objective_value": model.objective_value(),
        "status": str(model.status().exitstatus),
        "satisfiable": res
    }

if __name__ == "__main__":
    from runexp import default_parser
    parser = default_parser()

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.loads(f.read())
        runner = eval(args.runner)(func=eval(args.func),
                                   output=args.output,
                                   memory_limit=args.memory_limit,
                                   printlog=True)

        if args.unravel is True:
            runner.run_batch(config, parallel=args.parallel, num_workers=args.num_workers)
        else:
            runner.run_one(config)