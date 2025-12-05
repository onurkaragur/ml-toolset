from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

def build_single_model(name: str, params: dict):
    if name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    if name == "Multilayer Perceptron (MLP)":
        return MLPClassifier(**params)
    if name == "Perceptron":
        return Perceptron(**params)
    
    raise ValueError("Unknown Model Name: " + name)

def default_compare_model(name: str, seed: int = 42):
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=seed)
    
    if name == "Multilayer Perceptron (MLP)":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation="relu", random_state=seed)
    
    if name == "Perceptron":
        # eta0 = learning rate
        return Perceptron(max_iter=300, eta0=0.1, random_state=seed)
    
    raise ValueError("Unknown model name: " + name)