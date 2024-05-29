from Node import Node
import graphviz
import numpy as np

def print_if_then_rules(node, indent=""):
    if not node:
        return
    if node.is_leaf():
        probas = node.node_probas(node.df)
        class_index = np.argmax(probas)
        class_label = node.label_encoder.inverse_transform([class_index])[0]  # 실제 클래스 레이블로 변환
        print(indent + f"Then class = {class_label} with probability {probas[class_index]:.2f}")
    else:
        feature_name = node.feature_names[node.split_feature]
        print(indent + f"If {feature_name} <= {node.split_value}")
        print_if_then_rules(node.left, indent + "  ")
        print(indent + "Else")
        print_if_then_rules(node.right, indent + "  ")

def generate_dot(node, dot=None, parent=None, edge_label=""):
    if dot is None:
        dot = graphviz.Digraph()
        dot.node(name=str(node), label=str(node))
    if node.is_leaf():
        probas = node.node_probas(node.df)
        class_index = np.argmax(probas)
        class_label = node.label_encoder.inverse_transform([class_index])[0]  # 실제 클래스 레이블로 변환
        label = f"Class: {class_label}\nProba: {probas[class_index]:.2f}"
        dot.node(name=str(node), label=label)
    else:
        feature_name = node.feature_names[node.split_feature]
        dot.node(name=str(node), label=f"{feature_name} <= {node.split_value}")
        if node.left:
            dot = generate_dot(node.left, dot, str(node), "True")
        if node.right:
            dot = generate_dot(node.right, dot, str(node), "False")
    if parent:
        dot.edge(parent, str(node), label=edge_label)
    return dot