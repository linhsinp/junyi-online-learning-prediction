from enum import Enum


class ModelType(Enum):
    DecisionTree = "DecisionTreeClassifier"
    GradientBoosting = "GradientBoostingClassifier"
    LogisticL2 = "LogisticRegression_L2"
    LogisticL1 = "LogisticRegression_L1"
