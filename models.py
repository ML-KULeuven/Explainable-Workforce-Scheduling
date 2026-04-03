
import cpmpy as cp
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.expressions.variables import _BoolVarImpl
from cpmpy.transformations.normalize import toplevel_list
import pandas as pd
from utils import read_instance, LexicoSolver, plot_task, UNALLOC

# from utils import TEAMS

class DispersionFormulation:
    MAX_MINUS_MIN = "max-minus-min"
    MAX_DIFF = "max-diff",
    PROXY_MIN_MAX = "proxy-min-max",
    PROXY_MAX_MIN = "proxy-max-min",
    PROXY_SUM = "proxy-sum",


def allocation_model_file(json_file, **model_kwargs):
    tasks, calendars, same_allocation = read_instance(json_file)
    return AllocationModel(tasks, calendars, same_allocation, **model_kwargs)

class AllocationModel(LexicoSolver):


    def __init__(self, tasks, calendars, same_allocation, allow_unalloc=False, 
                 dispersion_formulation=DispersionFormulation.MAX_MINUS_MIN, 
                 break_symmetries=True,
                 make_time_worked_vars=True):
        super().__init__()

        self.TEAMS = sorted(set().union(*[t['team_ids'] for _, t in tasks.iterrows()]))

        self.tasks = tasks
        self.calendars = calendars
        self.same_alloc_groups = same_allocation
        self.allow_unalloc = allow_unalloc
        self.dispersion_formulation = dispersion_formulation

        # make variables
        self.alloc = cp.boolvar(shape=(len(tasks), len(self.TEAMS)), name="x")
        self.used = cp.boolvar(shape=len(self.TEAMS), name="used")
        if make_time_worked_vars:
            self.time_worked = cp.intvar(0, tasks['duration'].sum(), shape=len(self.TEAMS), name="time")

        self.soft, self.hard = [], []
        
        self.add_soft_constraint(self.task_is_allocated(allow_unalloc))
        self.add_soft_constraint(self.same_allocation())
        self.add_soft_constraint(self.task_team_compatibility())
        self.add_soft_constraint(self.overlapping_tasks())
        
        self.add_hard_constraint(self.team_usage())
        if make_time_worked_vars:
            self.add_hard_constraint(self.time_worked_constraints())

        if break_symmetries:
            self.add_hard_constraint(self.get_symmetry_breaking_constraints())

        self.disruptions = pd.DataFrame(columns=["team_id", "start_unavailable", "end_unavailable"])


    def add_soft_constraint(self, cons):
        self.soft += toplevel_list(cons)
        self += cons

    def add_hard_constraint(self, cons):
        self.hard += toplevel_list(cons)
        self += cons


    def task_is_allocated(self, allow_unalloc): # each task is assigned one team
        """
            Each task is assigned one team.
        """
        constraints = []
        for i, allocated in enumerate(self.alloc):
            if allow_unalloc:
                cons = cp.sum(allocated) <= 1
            else:
                cons = cp.sum(allocated) == 1
            cons.set_description(f"Task {i} should be allocated to a team")
            constraints.append(cons)

        return constraints
    
    def task_team_compatibility(self):
        """
            Some tasks cannot be assigned to certain teams.
        """
        constraints = []
        for idx, (_, task) in enumerate(self.tasks.iterrows()):
            for team in set(self.TEAMS) - task['team_ids']:
                cons = self.alloc[idx, self.TEAMS.index(team)] <= 0
                cons.set_description(f"Team {team} cannot be assigned to task {idx}")
                constraints.append(cons)
        return constraints

    def overlapping_tasks(self):
        """
            Overlapping tasks cannot be assigned to the same team.
        """
        constraints = []
        for idx, (_, task) in enumerate(self.tasks.iterrows()):
            overlapping = (self.tasks['original_start'] <= task['original_start']) & (self.tasks['original_end'] > task['original_start'])
            for team_idx, team in enumerate(self.TEAMS):
                # disable team for task if calendar is overlapping
                if len(self.calendars) > 0:
                    cal = self.calendars[self.calendars['team_id'] == team]
                    cal_overlapping = (cal['start_unavailable'] <= task['original_start']) & (cal['end_unavailable'] > task['original_start'])
                    if cal_overlapping.any():
                        cons = self.alloc[idx, team_idx] <= 0
                        cons.set_description(f"Team {team} is not available for task {idx}")
                        constraints.append(cons)
                
                # team cannot do overlapping tasks
                cons = cp.sum(self.alloc[overlapping, team_idx]) <= 1
                cons.set_description(f"Tasks {np.where(overlapping)[0]} cannot be assigned to the same team")
                constraints.append(cons)
        
        return constraints
    
    def same_allocation(self):
        """
            Some tasks must be assigned to the same team.
        """
        constraints = []
        for group in self.same_alloc_groups:
            group = self.tasks.index[self.tasks['task_id'].isin(group)].tolist()
            for idx, team in enumerate(self.TEAMS):
                cons = cp.AllEqual(self.alloc[group, self.TEAMS.index(team)])
                cons.set_description(f"Tasks {group} must be assigned to the same team (team {team})")
                constraints.append(cons)
        return constraints
    
    def team_usage(self):
        """
            Define the auxilary "used" variables that are true if a team is used.
        """
        constraints = []
        for i, team_is_used in enumerate(self.used):
            cons = cp.any(self.alloc[:,i]).implies(team_is_used)
            cons.set_description(f"Team {self.TEAMS[i]} is used if it is assigned to a task")
            constraints.append(cons)

        return constraints
    
    def time_worked_constraints(self):
        """
            Define the time worked for each team, only if it is used.
        """
        constraints = []
        for i, team_is_used in enumerate(self.used):
            alloc_time = cp.sum(self.alloc[:,i] * self.tasks['duration'])
            cons = team_is_used.implies(alloc_time == self.time_worked[i])
            cons.set_description(f"Definition of time worked for team {self.TEAMS[i]}")
            constraints.append(cons)
        return constraints

    def get_nb_teams_objective(self):
        """
            Minimize the number of teams used.
        """
        return cp.sum(self.used)
    
    def get_equivalent_teams(self):
        """
            Get the equivalent teams.
            Returns a list of lists of equivalent teams.
            Teams in the same group are equivalent.
        """
        teams_tasks = {team : set() for team in self.TEAMS}
        for i, task in self.tasks.iterrows():
            for team in task['team_ids']:
        
                if len(self.calendars) > 0:
                    cal = self.calendars[self.calendars['team_id'] == team]
                    cal_overlapping = (cal['start_unavailable'] <= task['original_start']) & (cal['end_unavailable'] > task['original_start'])
                    if cal_overlapping.any():
                        continue # cannot be assigned to this team due to calendar
                teams_tasks[team].add(i)
        
        # team_tasks[team] = set of tasks that CAN be assigned to team `team`
        # now find teams which can be assigned to the same sets.
        # check for equivalency
        groups = dict()
        for team, tasks in teams_tasks.items():
            if frozenset(tasks) not in groups:
                groups[frozenset(tasks)] = []
            groups[frozenset(tasks)].append(team)

        return list(groups.values())

    
    def get_symmetry_breaking_constraints(self):
        
        cons = []
        for group in self.get_equivalent_teams():
            # sort teams and set lex-leader constraints
            teams = sorted(group)
            for t1, t2 in zip(teams[:-1], teams[1:]):
                cons.append(self.used[self.TEAMS.index(t2)].implies(self.used[self.TEAMS.index(t1)]))
        return cons
            
    
    def get_dispersion_objective(self):
        """
            Minimize the dispersion of the time worked (max(time_worked) - min(time_worked))
            Formulation can be chosen from DispersionFormulation enum.
        """
        if self.dispersion_formulation == DispersionFormulation.MAX_MINUS_MIN:
            return cp.max(self.time_worked) - cp.min(self.time_worked)
        elif self.dispersion_formulation == DispersionFormulation.MAX_DIFF:
            return cp.max([x - y for x in self.time_worked for y in self.time_worked])
        elif self.dispersion_formulation == DispersionFormulation.PROXY_MIN_MAX:
            return cp.max(self.time_worked)
        elif self.dispersion_formulation == DispersionFormulation.PROXY_MAX_MIN:
            return -cp.min(self.time_worked)
        elif self.dispersion_formulation == DispersionFormulation.PROXY_SUM:
            return cp.sum(self.time_worked) # TODO: check if this is correct        
        else:
            raise ValueError(f"Unknown dispersion formulation: {self.dispersion_formulation}")
    
    def get_dispersion_value(self):
        """
            Get the dispersion value of the last solve call.
        """
        # assert self.status().exitstatus != ExitStatus.NOT_RUN, "Model should be solved before getting dispersion value"
        used = self.used.value()
        time_worked = self.time_worked.value()
        return max([tw for is_used,tw in zip(used, time_worked) if is_used]) - min([tw for is_used,tw in zip(used, time_worked) if is_used])
    
    def get_solution(self): # return a solution as a dataframe
        """
            Get the solution of the last solve call as a dataframe.
        """
        solution = self.tasks[['task_id', 'original_start', 'original_end']].copy()
        assigned_teams = np.sum(self.alloc.value() * np.arange(1,len(self.TEAMS)+1), axis=1)
        solution['assigned_team'] = [self.TEAMS[idx-1] if idx > 0 else UNALLOC for idx in assigned_teams]
        solution['start'] = solution['original_start']
        solution['end'] = solution['original_end']
        return solution
    
    def add_disruption(self, disruption):
        """
            Add a disruption to the model.
        """

        if isinstance(disruption, list): # multiple disruptions
            for d in disruption:
                self.add_disruption(d)
        
        else: # single disruption
            print("Adding disruption", disruption)
            self.disruptions = pd.concat([self.disruptions, disruption])
            for _, row in disruption.iterrows():
                for task_idx, task in self.tasks.iterrows():
                    overlapping = (row['start_unavailable'] <= task['original_end']) & (task['original_start'] <= row['end_unavailable'])
                    if overlapping:
                        cons = self.alloc[task_idx, self.TEAMS.index(row['team_id'])] == False
                        cons.set_description(f"Team {row['team_id']} cannot execute task {task['task_id']} due to a disruption from {row['start_unavailable']} to {row['end_unavailable']}")
                        self.add_hard_constraint(cons)
    
    ###########################################################
    #                     Visualization                        #
    ###########################################################

    def visualize_solution(self, sol=None, figsize=(10, 6), fig=None, ax=None, teams=None, **kwargs):
        """
            Visualize the solution of the last solve call using matplotlib.
        """
        sol = self.get_solution() if sol is None else sol
        if teams is None:
            teams = sorted(set(sol['assigned_team']))
        if fig is None:
            assert ax is None, "ax should not be provided if fig is not provided"
            fig, ax = plt.subplots(figsize=figsize)
                        
        for _, task in sol.iterrows():
            index_team = len(teams) # virtual unallocated team.
            if task['assigned_team'] in teams:
                index_team = teams.index(task['assigned_team'])

            plot_task(ax, task['start'], task['end'], index_team, task['task_id'], **kwargs)

        # plot calendar intervals as darkened bars
        for _, cal in self.calendars.iterrows():
            if cal['team_id'] in teams:
                plot_task(ax, cal['start_unavailable'], cal['end_unavailable'], teams.index(cal['team_id']), facecolor="black")
        
        ax.set_xlabel("Time in minutes")
        ax.set_ylabel("Team")
        ax.set_title("Solution to the allocation problem")
        ax.set_yticks(list(range(len(teams)+1)), teams+["Unset"])
    
        return fig, ax
        
    def _visualize_disruptions(self, ax, teams):
        for _, disruption in self.disruptions.iterrows():
            if disruption["team_id"] in teams:
                plot_task(ax, disruption['start_unavailable'], disruption['end_unavailable'],
                          teams.index(disruption['team_id']),
                          facecolor='red', alpha=0.5,hatch="//", height=0.8)

    def visualize_solution_with_disruptions(self, sol=None, fig=None, ax=None):
        if sol is None:
            sol = self.get_solution()
        fig, ax = self.visualize_solution(sol=sol, fig=fig, ax=ax)
        teams = sorted(set(sol['assigned_team']))
        self._visualize_disruptions(ax, teams)
        ax.set_title("Original solution with disruptions")
        return fig, ax

    def highlight_tasks(self, tasks, sol=None,fig=None, ax=None):
        if sol is None:
            sol = self.get_solution()
        fig, ax = self.visualize_solution(sol=sol, fig=fig, ax=ax)
        teams = sorted(set(sol['assigned_team']))
        
        plotted_tasks = set() # avoid plotting the same task multiple times
        for task in tasks:
            if isinstance(task, int):
                task = sol.loc[task]
            elif isinstance(task, str):
                task = next(row for i, row in sol.iterrows() if row['task_id'] == task)
            elif isinstance(task, _BoolVarImpl):
                for task_idx, bvars in enumerate(self.alloc):
                    if task in set(bvars):
                        task = sol.loc[task_idx]
                        break
                else:
                    raise ValueError(f"Task {task} not found in solution")
            else:
                raise ValueError(f"Invalid task type: {type(task)}")
            
            if task['task_id'] not in plotted_tasks:
                plot_task(ax, task['original_start'], task['original_end'], teams.index(task['assigned_team']), task['task_id'], facecolor="blue", alpha=1)
                plotted_tasks.add(task['task_id'])

        self._visualize_disruptions(ax, teams)
        return fig, ax



class SchedulingModel(AllocationModel):

    def __init__(self, tasks, calendars, same_allocation, allow_unalloc=False, dispersion_formulation=DispersionFormulation.MAX_MINUS_MIN, break_symmetries=True):
        
        # make start and end variables
        self.start, self.end = [], []
        for i, task in tasks.iterrows():
            self.start += [cp.intvar(int(task['release_date']), int(task['due_date']), name=f"start[{i}]")]
            self.end +=   [cp.intvar(int(task['release_date']), int(task['due_date']), name=f"end[{i}]")]

        self.start = cp.cpm_array(self.start)
        self.end = cp.cpm_array(self.end)
        
        super().__init__(tasks, calendars, same_allocation, allow_unalloc, dispersion_formulation, break_symmetries)
        # super will call `self.overlapping_tasks()`
        self.add_soft_constraint(self.precedence_constraints())

    def get_lower_bound(self, add_to_model=False, **kwargs):
        """
            Get the lower bound of the model.
            Relax the problem by considering all teams equivalent, and compute the minimum number of equivalent teams.
        """
        assert "solver" not in kwargs or kwargs["solver"] != "ortools", f"Solver for lb computation is always ortools, but user provided {kwargs['solver']}"

        capacity = cp.intvar(0, len(self.TEAMS), name="nb_teams")
        lb_solver = cp.SolverLookup.get("ortools") # need access to lower bound in case solve did not finish
        lb_solver += cp.Cumulative(self.start, self.tasks['duration'].tolist(), self.end, [1] * len(self.tasks), capacity)
        lb_solver.minimize(capacity)

        res = lb_solver.solve()
        if res is False:
            lb = 0
        else:
            lb = int(lb_solver.ort_solver.BestObjectiveBound())

        if add_to_model:
            self.add_hard_constraint(self.get_nb_teams_objective() >= lb)

        return lb
    
    def get_equivalent_teams(self):
        return [[]]
        
    def overlapping_tasks(self): # override parent method, need to take into account variable start and end
        """
            Overlapping tasks cannot be assigned to the same team.
        """
        constraints = []
        for team_idx, team in enumerate(self.TEAMS):

            starts = self.start.tolist()
            dur    = self.tasks['duration'].tolist()
            ends   = self.end.tolist()
            # heights = self.alloc[:, team_idx].tolist() # sneaky, height is 1 task is allocated to the team and 0 otherwise

            # also add the dummy tasks that represent the calendar intervals
            if len(self.calendars) > 0:
                cal = self.calendars[self.calendars['team_id'] == team]
                starts  += cal['start_unavailable'].astype(int).tolist()
                dur     += (cal['end_unavailable'] - cal['start_unavailable']).astype(int).tolist()
                ends    += cal['end_unavailable'].astype(int).tolist()
                # heights += [1] * len(cal['start_unavailable'])
            constraints.append(cp.NoOverlapOptional(starts, dur, ends, self.alloc[:, team_idx]))

        return constraints

    def precedence_constraints(self):
        """
            Some tasks share a precedence relation.
        """
        constraints = []
        for tid, task in self.tasks.iterrows():
            for succ in task['successors']:
                succ_idx = self.tasks[self.tasks['task_id'] == succ].index[0]
                cons = self.end[tid] <= self.start[succ_idx]
                cons.set_description(f"Task {tid} must be finished before task {succ} starts")
                constraints.append(cons)
        return constraints
    
    def get_solution(self): # return a solution as a dataframe
        """
            Get the solution of the last solve call as a dataframe.
        """
        solution = self.tasks[['task_id', 'original_start', 'original_end']].copy()
        assigned_teams = np.sum(self.alloc.value() * np.arange(1,len(self.TEAMS)+1), axis=1)
        solution['assigned_team'] = [self.TEAMS[idx-1] if idx > 0 else "Unallocated" for idx in assigned_teams]
        solution['start'] = self.start.value()
        solution['end'] = self.end.value()
        return solution
    
    def visualize_solution(self, *args, **kwargs):
        fig, ax = super().visualize_solution(*args, **kwargs)
        ax.set_title("Solution to the scheduling problem")
        return fig, ax