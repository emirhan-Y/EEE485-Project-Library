"""
Data prepare script.
====================

Sends the unprocessed pdf files to the processor
"""
from util import create_dataset_from_signature_pdf
from util import log
import os

_LOGGER = log(True, True, True)
"""Module logger"""

if __name__ == '__main__':
    parent_abspath = os.path.abspath(".")  # main folder path

    raw_abspath = os.path.join(parent_abspath, 'raw')  # unprocessed data path
    for filename in os.listdir(raw_abspath):  # each pdf file in the unprocessed data path
        source_file_abspath = os.path.join(raw_abspath, filename)
        # checking if it is a file
        if os.path.isfile(source_file_abspath):  # if it is a file,
            _LOGGER.debug_message('data read', 'preparing ' + filename)
            # send the file to the parser
            pages = create_dataset_from_signature_pdf(source_file_abspath, parent_abspath, filename.split('.')[0])
