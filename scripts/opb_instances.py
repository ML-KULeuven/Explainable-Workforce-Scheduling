
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
    do_symmbreak = bool(sys.argv[2])
    tasks, calendars, same_allocation = read_instance(fname)

    model = AllocationModel(tasks, calendars, same_allocation, 
                            make_time_worked_vars=False, # ensure it is a 0-1 model, no integer vars
                            break_symmetries=do_symmbreak)
    
    model.minimize(model.get_nb_teams_objective())

    
    cp.SolverLookup.get("exact", model).native_model.printInput()