import numpy as np
import cv2
import os


def count_files(directory_path):
    """
    Counts the number of files in the specified directory.
    """
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    num_files = len(files)
    return num_files


def csv_prepare(n, class_lst):
    """
    Converts the image data into numpy arrays and then to design matrices.
    Returns the design matrix and the response matrix.
    Saves x and y data to a csv file under the data/excel directory.

    Parameters
    ----------
    n : Total number of image data to be used from each class.
    class_lst : List of the class names, which its data will be used in string format.
    (ex: ['gokce','omer','emir',...])

    Returns
    -------
    #matrix_x : Design Matrix (without the first column being all 1's)
    #matrix_y : Response Matrix
    Returns True if able to prepare csv files for n data, else returns False.
    """
    # Directory path
    directory_path = r"..\data\final\{}"

    for i in class_lst:
        new_directory_path = directory_path.format(i)
        num_files = count_files(new_directory_path)
        if num_files < n:
            print(f"Not enough data for class {i} in directory {new_directory_path}")
            return False

    # Base path
    base_path = r"..\data\final\{}\{}_{}.jpg"

    matrix_x = None
    matrix_y = None
    for i in class_lst:
        for j in range(n):  # Takes n data instances from each class.
            image_path = base_path.format(i, i, j)
            rgb_image_arr = cv2.imread(image_path)  # Read the image
            image_arr = cv2.cvtColor(rgb_image_arr, cv2.COLOR_BGR2GRAY)
            # Check if the image is successfully read
            if image_arr is not None:
                # Process the image (you can do any further processing here)
                image_vector = image_arr.reshape(1, -1)
                if matrix_x is None:
                    matrix_x = image_vector
                else:
                    matrix_x = np.vstack([matrix_x, image_vector])
                # Convert the string to a numpy array
                y_str = np.array([i])
                if matrix_y is None:
                    matrix_y = y_str
                else:
                    matrix_y = np.vstack([matrix_y, y_str])
            else:
                print(f"Image {image_path} not found or couldn't be read.")

    # Save the array to a CSV file
    np.savetxt(r"..\data\excel\data_x.csv", matrix_x, delimiter=',')
    np.savetxt(r"..\data\excel\data_y.csv", matrix_y, delimiter=',', fmt='%s')

    #return matrix_x, matrix_y
    return True

