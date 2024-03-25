"""
Data processor script.
======================

Given pdf files of the signatures, this module generates 126x50 jpg files of the signatures, and loads them to their
respective folders
"""
import os
import cv2
import numpy as np

from pdf2image import convert_from_path

from util import log

_LOGGER = log(True, True, True)
"""Module logger"""


def create_dataset_from_signature_pdf(source_file_abspath, parent_abspath, filename):
    """
    Read a given pdf file, and extract all signatures inside

    Parameters
    ----------
    source_file_abspath: Path
        Absolute path of the pdf file
    parent_abspath: Path
        Absolute path of the parent file
    filename: str
        Name of the pdf file
    """
    label = filename.split('_')[0]  # the labels of the data points are gathered via the naming convention
    temp_abspath = os.path.join(parent_abspath, 'temp')  # define the folder paths
    pre_abspath = os.path.join(parent_abspath, 'pre')
    final_abspath = os.path.join(parent_abspath, 'final', label)

    _empty_folder(temp_abspath, 'temp')  # empty the temporary folders ahead of process
    _empty_folder(pre_abspath, 'pre')
    # library function to convert each page of the pdf file to a jpg image
    pages = convert_from_path(pdf_path=source_file_abspath, dpi=500, output_folder=temp_abspath)
    for count, page in enumerate(pages):
        page.save(os.path.join(pre_abspath, f'{label}_{count}.jpg'), 'JPEG')  # save each page as jpg image
    _LOGGER.debug_message('data read', 'page enumeration complete')

    _empty_folder(temp_abspath, 'temp')  # prepare the temp folder for next loop

    for file in os.listdir(pre_abspath):  # read each page jpg file
        _LOGGER.debug_message('data read', 'started reading ' + file + ' data file')
        image = cv2.imread(os.path.join(pre_abspath, file))
        data_counter = len(os.listdir(final_abspath))  # count how many images of the label type exist

        # find the pixel intervals where the signatures reside
        row_intervals = []
        currently_red_row = True
        int_start = 0
        for row in range(4, len(image)):
            row_arr = image[row].astype(int)  # get a row of the image
            check_val = np.sum(row_arr[:, 2] - row_arr[:, 1] - row_arr[:, 0])  # check if the row is dominantly red
            if check_val < 0:  # if row not red,
                if currently_red_row:  # and it was red previously,
                    int_start = row  # means we have entered a white zone, which is the start point of an interval
                currently_red_row = False  # we are not red now
            else:  # if row is red
                if not currently_red_row:  # but the previous row wasn't red,
                    int_end = row  # means we are entering a red zone, which is the end point of an interval
                    row_intervals.append([int_start, int_end])  # the interval ended, append it to the interval list
                currently_red_row = True  # we are now red
        if len(row_intervals) != 0:  # if at least one interval was found, do the same process to the columns
            col_intervals = []
            currently_red_col = True
            int_start = 0
            for col in range(4, len(image[0])):
                col_arr = image[:, col].astype(int)  # get a col of the image
                check_val = np.sum(col_arr[:, 2] - col_arr[:, 1] - col_arr[:, 0])  # check if the col is dominantly red
                if check_val < 0:  # if col not red,
                    if currently_red_col:  # and it was red previously,
                        int_start = col  # means we have entered a white zone, which is the start point of an interval
                    currently_red_col = False  # we are not red now
                else:  # if row is red
                    if not currently_red_col:  # but the previous col wasn't red,
                        int_end = col  # means we are entering a red zone, which is the end point of an interval
                        col_intervals.append([int_start, int_end])  # the interval ended, append it to the interval list
                    currently_red_col = True  # we are now red
            for row_int in row_intervals:  # for each row interval
                for col_int in col_intervals:  # and for each col interval,
                    # crop the image according to the row and col intervals
                    data_instance = image[row_int[0]:row_int[1], col_int[0]:col_int[1]]
                    # resize the image to 126 x 50
                    data_instance = cv2.resize(data_instance, (126, 50), interpolation=cv2.INTER_AREA)
                    valid = False  # check the validity of the image: if it is all white, it is empty, so discard
                    for row in range(len(data_instance)):
                        for col in range(len(data_instance[0])):
                            if np.sum(data_instance[row][col]) < 255:  # check if the pixel is reasonably dim
                                valid = True  # if dim pixel exists, the image is valid
                                break
                    if valid:  # save each valid image, which are the signature images
                        data_instance_grayscale = cv2.cvtColor(data_instance, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(os.path.join(final_abspath, f'{label}_{data_counter}.jpg'), data_instance_grayscale)
                        data_counter += 1
    _empty_folder(pre_abspath, 'pre')  # empty the pre folder ahead of the next iteration


def _empty_folder(abspath, folder_name):
    """
    Utility function to delete every file inside a given folder

    Parameters
    ----------
    abspath: Path
        Absolute path to the folder
    folder_name: str
        Name of the folder
    """
    for file in os.listdir(abspath):
        os.remove(os.path.join(abspath, file))
    _LOGGER.debug_message('data read', folder_name + ' folder emptied')
