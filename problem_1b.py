import numpy as np

# First defining a small number for float comparisons
EPSILON = 1e-9

def solve_single_constraint_lp(c, A_row, b_val, problem_type="min", debug=False):
    """
    Solves a linear programming problem with a SINGLE equality constraint
    using a simplified Simplex Tableau method (simpler version of Wikipedia explanation)

    Form: max/min c*x subject to A_row*x = b_val, x >= 0.

    Inputs:
        c (np.ndarray or list): 1D array/list of objective function coefficients
        A_row (np.ndarray or list): 1D array/list of the single constraint coefficients.
        b_val (float): The single constraint right-hand side value (must be >= 0).
        problem_type (str): "max" or "min" (default: "min").
        debug (bool): true = prints iteration details, just a bunch of work to ensure that it runs right (for me)

    Outputs (hopefully):
        tuple: (solution, objective_value)
               solution (a numpy array): Optimal variable values.
               objective_value (float): Optimal objective function value.
               Note: should return (None, None) if problem is unbounded.
    """

    # --- Input Validation and Setup ---
    c = np.array(c, dtype=float)
    A_row = np.array(A_row, dtype=float)
    b_val = float(b_val)

    num_vars = len(c)

    # If minimizing, convert to maximization internally
    c_orig = c.copy() # Keep original for final objective calculation if min
    if problem_type.lower() == "min":
        c = -c

    # --- Find Initial Basic Variable ---
    # Find the first variable with a positive coefficient in the constraint
    initial_basic_idx = -1
    for i in range(num_vars):
        if A_row[i] > EPSILON:
            initial_basic_idx = i
            break

    if initial_basic_idx == -1:
        # This case implies all A_row coefficients are <= 0.
        # If b_val > 0, the problem is infeasible.
        # If b_val = 0, only x=0 is possible if all A_row coeffs are 0,
        # or multiple solutions if some A_row coeffs are negative.
        # For simplicity under the assumption b>=0 and at least one ai>0:
        raise ValueError("Cannot find an initial basic variable (no positive coefficient in A_row).")

    if debug:
        print(f"INFO: Choosing x{initial_basic_idx+1} as initial basic variable.")

    # --- Build and Normalize Initial Tableau ---
    # Tableau: 2 rows (obj, constraint), N+1 columns (vars + RHS)
    tableau = np.zeros((2, num_vars + 1))

    # Fill constraint row (row 1) and normalize it
    pivot_val_initial = A_row[initial_basic_idx]
    tableau[1, :num_vars] = A_row / pivot_val_initial
    tableau[1, -1] = b_val / pivot_val_initial
    # Ensure the basic variable column has exactly 1 after division

    
    #tableau[1, initial_basic_idx] = 1.0

    # Fill objective row (row 0) - initially [-c | 0]
    tableau[0, :num_vars] = -c
    tableau[0, -1] = 0.0

    # Adjust objective row to make coefficient for basic variable zero
    multiplier = tableau[0, initial_basic_idx] # / tableau[1, initial_basic_idx] which is 1
    tableau[0, :] -= multiplier * tableau[1, :]
    # Ensure the basic variable coefficient is exactly zero
    
    #tableau[0, initial_basic_idx] = 0.0

    # Track the current basic variable index
    basic_var_idx = initial_basic_idx

    # --- Simplex Iterations ---
    iteration = 0
    max_iterations = 50 # Nothing should get past this at this level of problem, just a debug thing

    while iteration < max_iterations:
        if debug:
            print(f"\n--- Iteration {iteration} ---")
            var_labels = [f"x{i+1}" for i in range(num_vars)] + ["RHS"]
            basis_label = f"x{basic_var_idx+1}"
            print(f"Tableau (Basic Var: {basis_label}):")
            # Simple printout
            print("      " + " | ".join(f"{h:>6}" for h in var_labels))
            print(" z' | " + " | ".join(f"{v:>6.2f}" for v in tableau[0]))
            print(f" {basis_label:<3}| " + " | ".join(f"{v:>6.2f}" for v in tableau[1]))

        # 1. Check for Optimality (Maximization problem)
        obj_coeffs = tableau[0, :num_vars]
        all_non_negative = True
        for i in range(num_vars):
            if i != basic_var_idx and obj_coeffs[i] < -EPSILON:
                all_non_negative = False
                break
        if all_non_negative:
            if debug: print("\nOptimal solution found.")
            break

        # 2. Find Pivot Column (Entering Variable)
        # Most negative coefficient among non-basic variables
        pivot_col = -1
        min_coeff = -EPSILON
        for j in range(num_vars):
            if j != basic_var_idx and obj_coeffs[j] < min_coeff:
                min_coeff = obj_coeffs[j]
                pivot_col = j

        if pivot_col == -1:
            # Just for debugging saftey, this...theoretically should never be reached?
            break

        if debug:
            print(f"Pivot column (entering var): x{pivot_col+1} (index {pivot_col}) with coeff {obj_coeffs[pivot_col]:.2f}")

        # 3. Minimum Ratio Test (Find Leaving Variable / Pivot Row)
        # With only one constraint, the leaving variable is always the current basic variable.
        # We just need to check if the problem is unbounded.
        pivot_row = 1 # Always the constraint row
        entry_in_pivot_col = tableau[pivot_row, pivot_col]

        if entry_in_pivot_col <= EPSILON:
            # If the entry is non-positive, we can increase the entering
            # variable indefinitely without violating the constraint (since b>=0).
            print("Error: Problem is unbounded.")
            return None, None

        pivot_element = entry_in_pivot_col
        if debug:
            print(f"Pivot element (Row {pivot_row}, Col {pivot_col}): {pivot_element:.2f}")
            print(f"Leaving variable: x{basic_var_idx+1} (index {basic_var_idx})")


        # 4. Pivoting
        # a) Normalize the pivot row (row 1)
        tableau[pivot_row, :] /= pivot_element
        tableau[pivot_row, pivot_col] = 1.0 # Just making sure this is exactly 1

        # b) Eliminate entry in objective row (row 0)
        multiplier = tableau[0, pivot_col]
        tableau[0, :] -= multiplier * tableau[pivot_row, :]
        tableau[0, pivot_col] = 0.0 # same but 0

        # 5. Update Basis
        basic_var_idx = pivot_col # The entering variable becomes basic

        iteration += 1

    # --- End of Loop ---
    if iteration >= max_iterations:
        return None, None

    # --- Get Solution ---
    solution = np.zeros(num_vars)
    # The basic variable gets the right side value from the constraint row
    solution[basic_var_idx] = tableau[1, -1]

    # Get rid of small floating point errors 
    solution[np.abs(solution) < EPSILON] = 0.0

    # Final objective value (for the maximization problem)
    max_obj_value = tableau[0, -1]

    # If original problem was minimization, return negative of max value
    final_obj_value = -max_obj_value if problem_type.lower() == "min" else max_obj_value

    return solution, final_obj_value


