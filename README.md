# Pipe Classification

Validation of a fuzzy model  
Optimization of fuzzy model  
Exploration of alternate models  

### Dependencies

 - numpy
 - scipy

### Use

model parameters are defined in /models, model structure is defined in fuzzy_logic in a function corresponding to the model name.

fuzzylogic.py contains legacy parameters for models under functions denoted by static

run 

	python fuzzylogic.py

to test the model, run 

	optimize.py 

to optimize the parameters and save them to a new model file. 
Current optimization reduces errors from a rate of 0.51 to a rate of 0.12


