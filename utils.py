import time
import pandas as pd
from datetime import datetime

import numpy as np
import cpmpy as cp
from cpmpy.expressions.variables import NDVarArray

UNALLOC = "None"

import json
def read_instance(json_file):
    """
        Reads an instance from a json file and returns the data as
        - tasks: a dataframe with the task information with columns:
            [id, original_start, original_end, duration, team_ids]
        - teams: a dataframe with the team calendars with columns:
            [id, start_unavailable, end_unavailable]
        - same_allocation: a list of sets of task ids that must be allocated to the same team
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    teams = set(data['teams'])

    # process data
    task_data = []
    for tid in data['tasks']:
        td = dict(
            task_id = tid,
            duration = data['tasks_data'][tid]['duration'],
            team_ids = set(data['compatible_teams'].get(tid, teams)),
            original_start = data['original_start'][tid],
            original_end = data['original_end'][tid],
            release_date = data['start_window'][tid][0],
            due_date = data['end_window'][tid][1],
            successors = set(data['successors'].get(tid, set()))
        )
        task_data.append(td)

    task_df = pd.DataFrame(task_data)
    horizon = (min(task_df['release_date']), max(task_df['due_date']))

    teams_data = []
    for tid in teams:
        window = horizon[0]
        for s,e in sorted(data['calendar'][tid], key=lambda x: x[0]):
            e = min(e, horizon[1])
            if s > window:
                teams_data.append(dict(
                    team_id=tid,
                    start_unavailable=window,
                    end_unavailable=s
                ))
            window = e
            
            
        # add last interval to end of horizon
        if window < horizon[1]:
            teams_data.append(dict(
                team_id=tid,
                start_unavailable=window,
                end_unavailable=horizon[1]
            ))

    teams_df = pd.DataFrame(teams_data)

    same_allocation = [set(x) for x in data['same_allocation']]
    # some groups can be merged
    same_alloc_merged = []
    for group in same_allocation:
        for grp in same_alloc_merged:
            if grp & group:
                grp.update(group)
                break
        else:
            same_alloc_merged.append(set(group))

    return task_df, teams_df, same_alloc_merged


class LexicoSolver(cp.Model):


    def set_lexico_objectives(self, objectives, minimize=True):

        if isinstance(minimize, bool):
            minimize = [minimize] * len(objectives)

        self.lexico_objectives = list(zip(objectives, minimize))

    def solve(self, time_limit=None, **kwargs):
        if not hasattr(self, "lexico_objectives"): # default to single objective
            return super().solve(time_limit=time_limit, **kwargs)
        
        self.solver_stati = []
        for obj, minimize in self.lexico_objectives:
            self.objective(obj, minimize=minimize)
            res = super().solve(time_limit=time_limit, **kwargs)
            self.solver_stati.append(self.status())
            print("Objective value:", obj.value())

            if res is False:
                return res
                        
            # add constraint fixing the previous objective to its optimal value
            if minimize:
                self += obj <= obj.value()
            else:
                self += obj >= obj.value()

        # return status of last solve call
        return res


###########################################################
#                 Visualization functions                 #
###########################################################

def plot_task(ax, start, end, team, task_id=None, height=0.9, **kwargs):
    task_id = None
    default = dict(
        facecolor="#008000",
        edgecolor="black",
        linewidth=0.5,
    )

    default.update(**kwargs)

    ax.broken_barh([(start, end-start)], (team-height/2, height), **default)
    if task_id is not None:
        ax.text(start+(end-start)/2, team+0.2, task_id, ha='center', va='center')