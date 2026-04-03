from runexp import Runner
import pandas as pd
import os
import json
import time

from models import AllocationModel, SchedulingModel
from utils import read_instance
import cpmpy as cp

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

        solver_kwargs = config['solver_kwargs']

        objective = config['objective']
        if objective == "nb_teams":
            objective = model.get_nb_teams_objective()
        elif objective == "dispersion":
            nb_teams_objective = model.get_nb_teams_objective()
            dispersion_objective = model.get_dispersion_objective()
            objective = nb_teams_objective * dispersion_objective.get_bounds()[1] + dispersion_objective
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        model.minimize(objective)

        timings = dict()
        if isinstance(model, SchedulingModel) and config['compute_lb'] is True:
            t0 = time.time()
            lb = model.get_lower_bound(add_to_model=True, 
                                       num_workers=solver_kwargs['num_workers'], 
                                       time_limit=60)
            t1 = time.time()
            timings["lb_computation_time"] = t1 - t0
            timings["lb_value"] = lb

        return {
            "model": model,
            "solver_kwargs": dict(solver_kwargs),
            **timings
        }
    def description(self, config):
        return f"Solving {config['instance'].split('/')[-1]} with " + ",".join(f"{k}={v}" for k, v in config['model_kwargs'].items()) + f" using {config['solver_kwargs']['solver']}"

from cpmpy.solvers.ortools import OrtSolutionPrinter
def solve_model(model, solver_kwargs, **timings):

    solver_name = solver_kwargs.pop("solver")

    t0 = time.time()
    solver = cp.SolverLookup.get(solver_name, model)
    objective_trace = []
    def save_objectives():
        nb_teams= int(sum(model.used.value()))
        dispersion = int(model.get_dispersion_value())
        objective_trace.append(
            (time.time()-t0, nb_teams, dispersion)
        )

    if solver_name == "ortools":
        solver_kwargs['solution_callback'] = OrtSolutionPrinter(solver, display=save_objectives)

    res = solver.solve(**solver_kwargs)
    t1 = time.time()
    return {
        "wallclock_time": t1 - t0,
        "solver_time": model.status().runtime,
        "objective_value": model.objective_value(),
        "status": str(model.status().exitstatus),
        "satisfiable": res,
        "objective_trace": objective_trace,
        **timings
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