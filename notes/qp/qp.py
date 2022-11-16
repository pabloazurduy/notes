# Import packages.
import cvxpy as cp
import numpy as np

# Generate data.
np.random.seed(1)
n_partners = 100
score = np.random.rand(n_partners)

# Define and solve the CVXPY problem.
n_people = 20
x = cp.Variable((n_people,n_partners ), boolean=True)
mean_score = score.mean()*n_people

cost = cp.sum_squares(mean_score-x @ score)
prob = cp.Problem(cp.Minimize(cost))
prob.solve(verbose=True) # use scip install using https://www.cvxpy.org/install/#install-with-scip-support https://www.cvxpy.org/examples/basic/mixed_integer_quadratic_program.html#mixed-integer-quadratic-program

# Print result.
print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
score_by_person = np.sum(x.value * score, axis=1)
print(f'score by person {score_by_person}')