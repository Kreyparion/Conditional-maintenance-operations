import numpy as np
from random import choices

from torch import threshold
from project.env.wearing_functions import discrete_wearing_function




print(discrete_wearing_function()[1](5,4))