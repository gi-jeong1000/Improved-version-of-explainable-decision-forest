import numpy as np
from scipy.stats import entropy


class Node():
    def __init__(self, mask, feature_names=None, label_encoder=None):
        self.mask = mask
        self.df = None  # To store df for probability calculations
        self.feature_names = feature_names
        self.label_encoder = label_encoder

    def is_leaf(self):
        return self.left is None and self.right is None

    def split(self, df, feature_names=None):
        self.df = df  # Store df in the node for later use
        if feature_names is not None:
            self.feature_names = feature_names
        if np.sum(self.mask) == 1:
            self.left = None
            self.right = None
            return

        self.features = [int(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)]
        self.split_feature, self.split_value = self.select_split_feature(df)
        self.create_mask(df)

        if not self.is_splitable():
            self.left = None
            self.right = None
            return

        self.left = Node(list(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))),
                         self.feature_names, self.label_encoder)
        self.right = Node(list(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))),
                          self.feature_names, self.label_encoder)
        self.left.split(df)
        self.right.split(df)

    def is_splitable(self):
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == 0 or np.sum(
                np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == 0:
            return False
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == np.sum(
                self.mask) or np.sum(
                np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == np.sum(self.mask):
            return False
        return True

    def create_mask(self, df):
        self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_value
        self.right_mask = df[str(self.split_feature) + '_lower'] >= self.split_value
        self.both_mask = ((df[str(self.split_feature) + '_lower'] < self.split_value) & (
                    df[str(self.split_feature) + "_upper"] > self.split_value))

    def select_split_feature(self, df):
        feature_to_value = {}
        feature_to_metric = {}
        for feature in self.features:
            value, metric = self.check_feature_split_value(df, feature)
            feature_to_value[feature] = value
            feature_to_metric[feature] = metric
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature, feature_to_value[feature]

    def check_feature_split_value(self, df, feature):
        value_to_metric = {}
        values = list(set(list(df[str(feature) + '_upper'][self.mask]) + list(df[str(feature) + '_lower'][self.mask])))
        np.random.shuffle(values)
        values = values[:3]
        for value in values:
            left_mask = [True if upper <= value else False for upper in df[str(feature) + "_upper"]]
            right_mask = [True if lower >= value else False for lower in df[str(feature) + '_lower']]
            both_mask = [True if value < upper and value > lower else False for lower, upper in
                         zip(df[str(feature) + '_lower'], df[str(feature) + "_upper"])]
            value_to_metric[value] = self.get_value_metric(df, left_mask, right_mask, both_mask)
        val = min(value_to_metric, key=value_to_metric.get)
        return val, value_to_metric[val]

    def get_value_metric(self, df, left_mask, right_mask, both_mask):
        l_df_mask = np.logical_and(np.logical_or(left_mask, both_mask), self.mask)
        r_df_mask = np.logical_and(np.logical_or(right_mask, both_mask), self.mask)
        if np.sum(l_df_mask) == 0 or np.sum(r_df_mask) == 0:
            return np.inf
        l_entropy, r_entropy = self.calculate_entropy(df, l_df_mask), self.calculate_entropy(df, r_df_mask)
        l_prop = np.sum(l_df_mask) / len(l_df_mask)
        r_prop = np.sum(r_df_mask) / len(l_df_mask)
        return l_entropy * l_prop + r_entropy * r_prop

    def predict_probas_and_depth(self, inst, training_df):
        if self.is_leaf():
            return self.node_probas(training_df), 1
        if inst[self.split_feature] <= self.split_value:
            prediction, depth = self.left.predict_probas_and_depth(inst, training_df)
            return prediction, depth + 1
        else:
            prediction, depth = self.right.predict_probas_and_depth(inst, training_df)
            return prediction, depth + 1

    def node_probas(self, df):
        x = df['probas'][self.mask].mean()
        return x / x.sum()

    def get_node_prediction(self, training_df):
        v = training_df['probas'][self.mask][0]
        v = [i / np.sum(v) for i in v]
        return np.array(v)

    def opposite_col(self, s):
        if 'upper' in s:
            return s.replace('upper', 'lower')
        else:
            return s.replace('lower', 'upper')

    def calculate_entropy(self, test_df, test_df_mask):
        x = test_df['probas'][test_df_mask].mean()
        return entropy(x / x.sum())

    def count_depth(self):
        if self.right is None:
            return 1
        return max(self.left.count_depth(), self.right.count_depth()) + 1

    def number_of_children(self):
        if self.right is None:
            return 1
        return 1 + self.right.number_of_children() + self.left.number_of_children()

    def has_same_class(self, df):
        labels = set([np.argmax(l) for l in df['probas'][self.mask]])
        return len(labels) <= 1