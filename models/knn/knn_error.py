"""
KNN model error handlers
========================

classes
-------
* knn_error: manage error generated by the knn model
"""


class knn_error(Exception):
    def __init__(self, message):
        super().__init__(message)
