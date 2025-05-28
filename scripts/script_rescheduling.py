import pandas as pd

from generator import generate_disruption
import cpmpy as cp
import matplotlib.pyplot as plt
from models import AllocationModel, SchedulingModel
from explain import FeasibilityResotorationModel
from utils import read_instance, UNALLOC
import os
import numpy as np
this_folder = os.path.abspath(os.path.dirname(__file__))
root_folder = os.path.join(this_folder, "../")


def main_generate_disruption():
    instance = os.path.join(root_folder, f"data/anon_jsons/instance_50.json")
    tasks, calendars, same_allocation = read_instance(instance)
    model = AllocationModel(tasks, calendars, same_allocation)
    model.set_lexico_objectives([
        model.get_nb_teams_objective(),
        model.get_dispersion_objective()
    ])

    model.solve(time_limit=5)
    sol = model.get_solution()
    model.visualize_solution()
    used_teams = list(model.get_solution()['assigned_team'].unique())
    disruption = generate_disruption(
        num_disruptions=(1,1), num_teams_affected=(2,2), duration=(15,30),
        horizon=sol.end.max(), teams = used_teams
    )

    model = FeasibilityResotorationModel(tasks, calendars, same_allocation, disruption=disruption)
    model.set_original_solution(sol)

    model.visualize_original_solution_with_disruptions()

    model.set_lexico_objectives(*zip(*([
        (model.nb_of_done_tasks(), False),
        (model.nb_of_reallocated_tasks(), True),
        (model.nb_of_shifted_tasks(), True),
        (model.sum_of_shifted_tasks(), True),
        (model.max_of_shifted_tasks(), True)
    ])))

    model.solve(time_limit=5)
    model.visualize_repair()
    plt.show()


def main_scenario_file():
    def get_availability_slots(calendar_matrix: np.ndarray):
        availability_slots = []
        start_slot = None
        n = len(calendar_matrix)

        for i in range(n):
            if calendar_matrix[i] == 1 and start_slot is None:  # Start of an available block
                start_slot = i
            elif calendar_matrix[i] == 0 and start_slot is not None:  # End of an available block
                availability_slots.append((start_slot, i - 1))
                start_slot = None
        if start_slot is not None:  # If the array ends with an available block
            availability_slots.append((start_slot, n - 1))

        return availability_slots
    import json
    instance = os.path.join(root_folder, f"data/anon_scenarios/scenario_0.json")
    tasks, calendars, same_allocation = read_instance(instance)
    dict_instance = json.load(open(instance, "r"))
    disruptions = []
    teams = dict_instance["teams"]
    bin_calendar_disruption = {teams[i]: np.zeros(1000) for i in range(len(teams))}
    for st, end, i in dict_instance['disruption']["disruptions"]:
        bin_calendar_disruption[teams[i]][st:end] = 1
    for team in bin_calendar_disruption:
        slots = get_availability_slots(bin_calendar_disruption[team])
        for st, end in slots:
            disruptions.append(dict(team_id=dict_instance["teams"][teams.index(team)],
                                    start_unavailable=st,
                                    end_unavailable=end))
    disruptions = pd.DataFrame(disruptions)
    model = FeasibilityResotorationModel(tasks, calendars, None,
                                         disruption=disruptions)
    solution = model.tasks[['task_id', 'original_start', 'original_end']].copy()
    solution['assigned_team'] = [model.TEAMS[idx] for idx in dict_instance["base_solution"]["allocation"]]
    solution['start'] = solution['original_start']
    solution['end'] = solution['original_end']
    model.set_original_solution(solution)
    model.visualize_original_solution_with_disruptions()
    model.set_lexico_objectives(*zip(*([
        (model.nb_of_done_tasks(), False),
        (model.nb_of_reallocated_tasks(), True),
        (model.nb_of_shifted_tasks(), True),
        # (model.sum_of_shifted_tasks(), True),
        # (model.max_of_shifted_tasks(), True)
    ])))
    model.solve(time_limit=20, num_workers=10, log_search_progress=True)
    model.visualize_repair()
    plt.show()


if __name__ == "__main__":
    main_scenario_file()