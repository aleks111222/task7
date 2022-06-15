import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

xData = pd.read_csv("Testing.csv")

print(xData.columns.size())

model = BayesianModel([('itching', 'skin_rash'), ('itching', 'joint_pain'), ('joint_pain', 'scurring'), ('scurring', 'prognosis')])

model.fit(xData, estimator = MaximumLikelihoodEstimator)

infer = VariableElimination(model)
q = infer.query(variables = ['prognosis'], evidence = {'itching' : 1, 'cough' : 1})

print(q)