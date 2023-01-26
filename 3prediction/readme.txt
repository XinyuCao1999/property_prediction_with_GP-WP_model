1.ProductDesignFunction.py：
this file include two kinds of functions for calculating errors, which will be impoerted to model_vertify_in_paper.py

2.model_vertify_in_paper.py: 
all the models for 20 properties in paper can be vertified by running this file.
model 1:simple (SVR model in paper). model["simple_model_coef"] and model["simple_model_intercept"] is the coefficient.
model 2:normal GP(GP model in paper). 
model 3-5:GP-WP in paper.
if prior predictive checking result  is "non-zero", final model corresponds to model 5——conbination
if prior predictive checking result  is "zero", final model corresponds to model 3——warping function

3.new_molecule_prediction.ipynb
illustrate how to make prediction for a new molecule.