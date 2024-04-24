"""
PCA analysis definitions
=====================

classes
-------
* pca: applies pca to given data and holds relevant parameters
"""
import numpy as np
from numpy import linalg
from typing import Optional
from util import log


class pca:
    """
    Principal Component Analysis class
    """
    DEBUG_MSG_ENABLE = True
    WARNING_MSG_ENABLE = True
    ERROR_MSG_ENABLE = True
    _LOGGER = None

    def __init__(self, data_points: Optional[np.ndarray] = None, var_contained: Optional[float] = None,
                 predictors: int = -1):
        pca._LOGGER = log(pca.DEBUG_MSG_ENABLE, pca.WARNING_MSG_ENABLE, pca.ERROR_MSG_ENABLE)
        pca._LOGGER.debug_message('pca', 'pca initialization')
        self._data_points = data_points
        self._var_contained = var_contained
        self._required_eigen_values = predictors
        self._num_data_points = None
        self._num_features = None
        self._feature_mean = None
        self._feature_std = None
        self._data_points_normal = None
        self._cov_matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._eigen_vv_matrix = None

        self._new_basis = None
        self._new_dataset = None
        pca._LOGGER.debug_message('pca', 'data points: ' + (
            'Null' if data_points is None else 'non-Null') + ' variance contained: ' + (
                                      'Null' if var_contained is None else str(var_contained)))

    def analyze(self):
        """
        Analyze the data points given to the pca instance, according to the desired variance
        """
        if self._data_points is None:
            pca._LOGGER.error_message('pca', 'no data points were entered to the pca to be analyzed')
            raise RuntimeError('pca has no data to analyze!')
        self._num_data_points = len(self._data_points)
        pca._LOGGER.debug_message('pca', 'number of data points: ' + str(self._num_data_points))
        self._num_features = len(self._data_points[0])
        pca._LOGGER.debug_message('pca', 'number of features: ' + str(self._num_features))
        self._feature_mean = np.sum(self._data_points, axis=0) / self._num_data_points
        pca._LOGGER.debug_message('pca', 'mean of features calculated')
        self._feature_std = np.sqrt(
            np.sum((self._data_points - self._feature_mean) ** 2, axis=0) / self._num_data_points)
        self._feature_std[self._feature_std == 0] = 1
        pca._LOGGER.debug_message('pca', 'standard deviation of features calculated')
        self._data_points_normal = (self._data_points - self._feature_mean) / self._feature_std
        pca._LOGGER.debug_message('pca', 'normalized data points')
        self._cov_matrix = np.matmul(self._data_points_normal.transpose(),
                                     self._data_points_normal) / self._num_data_points
        cov_matrix_validity = np.allclose(self._cov_matrix, self._cov_matrix.transpose(), rtol=0, atol=0)
        if cov_matrix_validity:
            pca._LOGGER.debug_message('pca', 'covariance matrix calculated, valid')
        else:
            pca._LOGGER.warning_message('pca', 'covariance matrix calculated, invalid')
        self._eigenvalues, self._eigenvectors = linalg.eigh(self._cov_matrix)
        pca._LOGGER.debug_message('pca', 'eigenvalues and eigenvectors of covariance matrix were found')
        self._eigen_vv_matrix = np.zeros([len(self._data_points_normal[0]), len(self._data_points_normal[0]) + 1])
        self._eigen_vv_matrix[:, 0] = np.absolute(self._eigenvalues)
        self._eigen_vv_matrix[:, 1:] = self._eigenvectors.transpose()
        sorted_eigenvalue_indices = np.argsort(self._eigen_vv_matrix[:, 0])
        self._eigen_vv_matrix = self._eigen_vv_matrix[sorted_eigenvalue_indices[::-1]]
        pca._LOGGER.debug_message('pca', 'sorted eigenvalues and eigenvectors matrix is constructed')

        if self._required_eigen_values == -1:
            self._find_number_of_required_predictors()

        self._new_basis = self._eigen_vv_matrix[:self._required_eigen_values, 1:].transpose()
        pca._LOGGER.debug_message('pca', 'new basis is formed')
        self._new_dataset = np.matmul(self._data_points_normal, self._new_basis)  # Z = XU
        pca._LOGGER.debug_message('pca', 'the dataset is transformed')

    def _find_number_of_required_predictors(self):
        total_var = np.sum(self._eigen_vv_matrix[:, 0], axis=0)
        cur_var = 0
        self._required_eigen_values = 0
        while cur_var / total_var < self._var_contained and self._required_eigen_values < len(self._eigen_vv_matrix):
            self._required_eigen_values += 1
            cur_var = np.sum(self._eigen_vv_matrix[:self._required_eigen_values, 0], axis=0)
        pca._LOGGER.debug_message('pca', 'required amount of predictors is found: ' + str(self._required_eigen_values))

    def reconstruct_original_data(self):
        """
        Tries to reconstruct the original data points entered from the new dataset and basis

        Returns
        -------
        reconstructed: np.array
            Reconstructed version of the original dataset
        """
        reconstructed = np.matmul(self._new_dataset, self._new_basis.transpose())  # X_hat = ZU^T
        return reconstructed * self._feature_std + self._feature_mean  # denormalize the reconstruction

    def reform_data(self, data_points_transform: np.ndarray):
        """
        Converts a given data point to the new form

        Parameters
        ----------
        data_points_transform: np.array
            Data point in the original form

        Returns
        -------
        reformed: np.array
            Reformed version of the original data point
        """
        return np.matmul((data_points_transform - self._feature_mean) / self._feature_std, self._new_basis)

    def return_to_normal(self, data: np.ndarray):
        """
        Converts a given data point in the converted form, back to its original form

        Parameters
        ----------
        data: np.array
            Data point in the converted form

        Returns
        -------
        reformed: np.array
            Original version of the data point
        """
        normal = np.matmul(data, self._new_basis.transpose())  # X_hat = ZU^T
        return normal * self._feature_std + self._feature_mean  # denormalize the reconstruction

    def set_data_points(self, data_points):
        """
        Set the data points of the pca instance

        Parameters
        ----------
        data_points: np.array
            New data points of the pca instance
        """
        if self._data_points is not None:
            pca._LOGGER.warning_message('pca', 'data points lost as data_points was set while not empty')
        self._data_points = data_points
        pca._LOGGER.debug_message('pca', 'pca data points set')

    def set_variance_contained(self, variance_contained):
        """
        Set the desired ratio of variance wanted from the pca analysis

        Parameters
        ----------
        variance_contained: float
            The desired ratio of variance
        """
        self._var_contained = variance_contained
        pca._LOGGER.debug_message('pca', 'variance contained set')

    def get_data_points(self):
        return self._data_points

    def get_var_contained(self):
        return self._var_contained

    def get_num_data_points(self):
        return self._num_data_points

    def get_num_features(self):
        return self._num_features

    def get_feature_mean(self):
        return self._feature_mean

    def get_feature_std(self):
        return self._feature_std

    def get_data_points_normal(self):
        return self._data_points_normal

    def get_cov_matrix(self):
        return self._cov_matrix

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_eigenvectors(self):
        return self._eigenvectors

    def get_eigen_vv_matrix(self):
        return self._eigen_vv_matrix

    def get_required_eigenvalues(self):
        return self._required_eigen_values

    def get_new_basis(self):
        return self._new_basis

    def get_new_dataset(self):
        return self._new_dataset
