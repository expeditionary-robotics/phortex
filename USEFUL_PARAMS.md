# Useful Parameters

There are several useful global parameters that are embedded in the code base. 
Ideally, we couldn't have many of these, but we will seek to document the 
existing ones here:
* `KEEP_FEASIBLE`: Boolean that determines whether the trajectory optimizer is constrained to keep the solution feasible at each iteration of planning.
  * Location: `src/planning/fumes/fumes/planner/utils.py`
* `THRESH_BUDGET_LB`: Float between 0 and 1 that determines the lower bound on trajectory length. If this variable is set to 0.50, the planner is required to use at least 50% of the total budget.
  * Location: `src/planning/fumes/fumes/planner/utils.py`

