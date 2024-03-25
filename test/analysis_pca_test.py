"""
PCA test script
===============
* PCA: TEST: PASSED
"""

import os
import cv2

from analysis import pca
import numpy as np

if __name__ == '__main__':
    X = []
    data_paths = [os.path.abspath('../data/final/gokce'), os.path.abspath('../data/final/emir')]
    for data_path in data_paths:
        for data_instance in os.listdir(data_path):
            x = cv2.imread(os.path.join(data_path, data_instance), 0).flatten()
            X.append(x)
    principal_component_analysis = pca(np.array(X), 0.9)
    principal_component_analysis.analyze()

    reconstructed_X = principal_component_analysis.reconstruct_original_data()

    saveloc_abspath = os.path.join(os.path.abspath("."), 'pca')
    cnt = 0
    for data_path in data_paths:
        for data_instance in os.listdir(data_path):
            name = data_instance.split('.')[0]
            cv2.imwrite(os.path.join(os.path.join(saveloc_abspath, name.split('_')[0]), f'{name}_reconstruction.jpg'),
                        (np.rint(reconstructed_X[cnt].reshape(50, 126))).astype(int))
            cnt += 1

    vector_vis = np.zeros(principal_component_analysis.get_required_eigenvalues())
    vector_vis[0] = 1

    vis_first_comp = principal_component_analysis.return_to_normal(vector_vis).reshape(50, 126)
    cv2.imwrite(f'first_component.jpg', (np.rint(vis_first_comp).astype(int)))
