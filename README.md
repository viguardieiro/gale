# Globally Assessing Local Explanations (GALE)
GALE is a Python library used to assess the similarity of local explanations from methods such as LIME, SHAP or generalized additive models (GAMs). To do so, GALE models the relationship between the explanation space and the model predictions as a scalar function. Then, we compute the topological skeleton of this function. This topological skeleton acts as a signature, which we use to compare outputs from different explanation methods. 

