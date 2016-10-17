# Pipe Classification

Validation of a fuzzy model  
Optimization of fuzzy model  
Exploration of alternate models  

### Dependencies

 - numpy
 - scipy

### Description

**Model Representation**  
Each model's parameters are represented in a .csv file in the models/ directory. These take the the form of rows where each row is one bound for a variable, each variable has mutlitple bounds. The rows follow this format:

	variable,  grade, bounds_1, bouns_2, bounds_3, bounds_4(opt)
	
Where there are either 3 or 4 bounds depending on whether the bound is trapezoidel or triangular. A variable may have any combination of triangular and trapezoidal bounds. The grade is the value associated with this particular bound. The variable is the label for the input variable that the bound corresponds to. The model is represented in such a way so that parameters can be loaded, updated and saved.

**Bounds Implementation**  
The Fuzzy Logic for the bounds is implemented such that given an input, the model calculates how much that variable belongs to each of the bounds, and then returns a weighted average of all the bounds that the model belongs to. In the condition model there is first a layer of bounds that correspond to each of the input variables, and then a pooling bound that has the first layer bounds as it's unput.

**Training**  
The model paramaters are optimized using scipy's minimize function. The optimization method found to be the most effective given these parameters is [Broyden–Fletcher–Goldfarb–Shanno](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm) algorithm (BFGS). In order to perform parameter updates, the first and second derivatives are estimated after several iterations of training. Broadly, BFGS trains by iteratively updating the parameters based on how much a change to each parameter will impact the output of the model. Training uses the difference between the (floating point) score achieved by the model and the expected score as the loss function. Training Improves the accuracy of the bounds model (as compared to a rule-based if/else model) from an accuracy of 0.49 to 0.88.

### Use

model parameters are defined in /models, model structure is defined in fuzzy_logic in a function corresponding to the model name.

fuzzylogic.py contains legacy parameters for models under functions denoted by static

to test the model run 

	python fuzzylogic.py

to optimize the model run 

	optimize.py 

 and save to a new model file. 



