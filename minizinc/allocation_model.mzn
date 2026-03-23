% Worforce Allocation problem in Minizinc.
% Given a set of teams and tasks, allocate each task to exactly one team; tasks have a fixed start time.
% Teams have to be compatible with the tasks they are assigned to; 
%   they can only do a single task at a time and can only do tasks if they are available according to their calendar;
%   teams have to be compatible with the tasks they are assigned to.
% 
% By-level optimization problem with objectives:
% - minimize the number of teams used
% - minimize the unfairness between teams (difference in max/min time worked)
%
% Model written by Ignace Bleukx, loosely based on the CPMpy model for the paper:
%   Bleukx, I., Boumazouza, R., Guns, T., Laage, N. and Poveda, G.,
%   Modeling and explaining an industrial workforce allocation and scheduling problem. 
%   CP 2025, Leibniz International Proceedings in Informatics, 340.
%
% Contact: ignace.bleukx@kuleuven.be

include "globals.mzn";

enum Team;
enum Task;

int: horizon;
set of int: Time = 0..horizon;

array[Task] of int: duration;
array[Task] of int: released;
array[Task] of int: due;
array[Task, Team] of int: compatible;
array[int] of set of Task: same_allocation;
array[int] of record(Team: team, Time: start_unavailable, Time: dur_unavailable): calendars;

array[Task] of var Team: Alloc;  % Team allocated to each task

array[Task] of int: original_start;
array[Task] of Time : Start = [original_start[t] | t in Task]; % fix start time of each task

% Tasks in the same same_allocation group should be assigned to the same team
constraint forall(group in same_allocation)(
    all_equal([Alloc[task] | task in group])
);

% Teams should be compatible with the tasks they are assigned to
constraint forall(team in Team, task in Task)(
    compatible[task, team] == 0 -> (Alloc[task] != team)
);

% Tasks assigned to the same team should not overlap
% task-decomposition of no-overlap constraint
constraint forall(team in Team)(
    forall(task in Task)(
        alldifferent([Alloc[t2] | t2 in Task where Start[t2] <= Start[task] /\ Start[t2] + duration[t2] > Start[task]])
    )
);

% Tasks assigned to the same team should not overlap with the calendar of the team
constraint forall(team in Team, task in Task)(
    exists([cal.team == team /\ 
            cal.start_unavailable <= Start[task] /\ 
            cal.start_unavailable + cal.dur_unavailable > Start[task] | cal in calendars]) -> Alloc[task] != team
);

% A team is used if at least one task is assigned to it
array[Team] of var bool: Used = [exists([Alloc[task] == team | task in Task]) | team in Team];
% Time worked for each team, only if the team is used
array[Team] of var opt int: TimeWorked = [sum([duration[task] | task in Task where Alloc[task] == team]) | team in Team where Used[team]];

% Bi-objective optimization problem:
% - minimize the number of teams used
% - minimize the unfairness between teams (difference in max/min time worked)
solve minimize sum(duration) * nvalue(Alloc) 
              + (max(TimeWorked) - min(TimeWorked));

output [ "Number of used teams:", show(sum(Used)), 
        "\nAllocation: ", show(Alloc),
       "\nTime worked: ", show(TimeWorked), 
       "\nDispersion: ", show(max(TimeWorked) - min(TimeWorked))];