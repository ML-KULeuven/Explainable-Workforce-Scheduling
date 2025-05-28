
from models import SchedulingModel, DispersionFormulation
from utils import UNALLOC, LexicoSolver, plot_task
import pandas as pd

import cpmpy as cp

class FeasibilityResotorationModel(SchedulingModel):

    def __init__(self, tasks, calendars, same_allocation, dispersion_formulation=DispersionFormulation.MAX_MINUS_MIN, disruption=None):
        
        self.disruption = disruption
        calendars_and_disruption = pd.concat([calendars, disruption], ignore_index=True)
        
        super().__init__(tasks, calendars_and_disruption, same_allocation, allow_unalloc=True, dispersion_formulation=dispersion_formulation)
        self.calendars = calendars
    

    def set_original_solution(self, solution=None):
        self.original_solution = solution
        self.bool_solution = [[t == team for t in self.TEAMS] for team in solution['assigned_team']]
        self.bool_solution = cp.cpm_array(self.bool_solution)

        used_teams = set(self.original_solution['assigned_team'])

        for i, team in enumerate(self.TEAMS):
            if team not in used_teams:
                self += self.used[i] == 0

    ###########################################################
    #                     Similarity objectives               #
    ###########################################################

    def nb_of_done_tasks(self):
        return cp.sum(self.alloc)
        return cp.sum(cp.sum(row1) == cp.sum(row2) for row1, row2 in zip(self.alloc, self.bool_solution))
    
    def nb_of_reallocated_tasks(self):
        return cp.sum(self.alloc != self.bool_solution)

    def nb_of_shifted_tasks(self):
        return cp.sum(self.start != self.original_solution['start'].astype(int))

    def sum_of_shifted_tasks(self):
        return cp.sum(cp.abs(self.start - self.original_solution['start'].astype(int)))

    def max_of_shifted_tasks(self):
        return cp.max(cp.abs(self.start - self.original_solution['start'].astype(int)))
    


    ###########################################################
    #                     Visualization                        #
    ###########################################################

    def visualize_original_solution_with_disruptions(self, fig=None, ax=None):
        fig, ax = super().visualize_solution(sol=self.original_solution, fig=fig, ax=ax)
        teams = sorted(set(self.original_solution['assigned_team']))
        for _, disruption in self.disruption.iterrows():
            if disruption["team_id"] in teams:
                plot_task(ax, disruption['start_unavailable'], disruption['end_unavailable'],
                          teams.index(disruption['team_id']),
                          facecolor='red', alpha=0.5,hatch="//", height=0.8)
            
        ax.set_title("Original solution with disruptions")
        return fig, ax
    

    def visualize_repair(self):
        # TODO: not sure if this will work if a task is unallocated
        
        used_teams = sorted(set(self.original_solution['assigned_team']))

        # first plot the unchanged tasks
        idx_changed = (self.original_solution != self.get_solution()).any(axis=1)
        unchanged = self.original_solution[~idx_changed].copy()
        fig, ax = self.visualize_solution(sol=unchanged, facecolor="grey", alpha=0.7, teams=used_teams)

        # plot the changed tasks in their old position
        changed = self.original_solution[idx_changed].copy()
        fig, ax = self.visualize_solution(sol=changed, fig=fig, ax=ax, alpha=0.3, teams=used_teams)
        
        # plot the new solution
        repaired_sol = self.get_solution()
        fig, ax = self.visualize_solution(sol=repaired_sol[idx_changed], alpha=0.3, facecolor="blue", fig=fig, ax=ax, teams=used_teams) # new solution
                
        # add arrows to show to repair
        for i, orig_sol in changed.iterrows():
            new_sol = repaired_sol.loc[i]
            duration = orig_sol['end'] - orig_sol['start']
            index_orig_sol = len(used_teams)
            index_new_sol = len(used_teams)
            if orig_sol['assigned_team'] in used_teams:
                index_orig_sol = used_teams.index(orig_sol['assigned_team'])
            if new_sol['assigned_team'] in used_teams:
                index_new_sol = used_teams.index(new_sol["assigned_team"])

            ax.annotate("", xycoords='data',
                            xytext=(orig_sol['start']+duration//2, index_orig_sol),
                            xy=    (new_sol['start']+duration//2, index_new_sol),
                            arrowprops=dict(arrowstyle="->", edgecolor="red"),
            )

        for _, disruption in self.disruption.iterrows():
            if disruption["team_id"] in used_teams:
                plot_task(ax, disruption['start_unavailable'], disruption['end_unavailable'],
                          used_teams.index(disruption['team_id']),
                          facecolor='red', alpha=0.5, hatch="//", height=0.8)
            
        ax.set_title("Repaired solution using re-allocation and re-scheduling of tasks")

        
        return fig, ax