
import sys
import os

# Add parent directory to system path to import models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AllocationModel
from utils import read_instance

if __name__ == "__main__":

    import os
    import sys
    import cpmpy as cp

    fname = sys.argv[1]
    if sys.argv[2] == "true":
        do_symmbreak = True
    elif sys.argv[2] == "false":
        do_symmbreak = False
    else:
        raise ValueError("Invalid argument for symmetry breaking: %s" % sys.argv[2])
    
    tasks, calendars, same_allocation = read_instance(fname)

    model = AllocationModel(tasks, calendars, same_allocation, 
                            make_time_worked_vars=False, # ensure it is a 0-1 model, no integer vars
                            break_symmetries=do_symmbreak)
    
    model.minimize(model.get_nb_teams_objective())

    
    solver = cp.SolverLookup.get("exact", model)
    solver.native_model.printFormula()
    # solver.solve(time_limit=10)
    # print(solver.status())
    # solver.solve(model)