import os
from help import load_signature_data

train_x, train_y, test_x, test_y = load_signature_data(os.path.abspath('_data/final'), 42, test_percentage=0.2, hot_encode=False)
print('foo')