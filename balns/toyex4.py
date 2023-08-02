from pyscipopt import Model

J = range(10)
U = [10*j for j in J]  # upper bound on x


model = Model("Formulation3")

#
# variables
#
x = [model.addVar(vtype="C", lb=0, ub=U[j]) for j in J]

#
# objective
#
model.setObjective(sum(x), sense="maximize")

#
# constraints
#
# We use model.addConsCardinality
#    Add a cardinality constraint that allows at most 'cardval' many nonzero variables.
# Only the following two parameters are used:
#       :param consvars: list of variables to be included
#       :param cardval: nonnegative integer
#
model.addConsCardinality([x[3],x[7]],1)
#model.writeProblem("model6.cip")

list =[]
for var in model.getVars():
    list.append(var)

print(list, "list")

#
# solve
#
model.optimize()
# print solution
#
for j in J:
    print("j=%2s, x[j]=%5s" % (j,model.getVal(x[j])))