import numpy as np

from my_utils.sorting.itertion_insert_sorter import IterationInsertSorter

a = IterationInsertSorter(30)

a.add(data=np.array([10.]), top=0)
a.add(data=np.array([11.]), top=1)
a.add(data=np.array([10.5, 10.5]), top=2)
a.add(data=np.array([9., 9.]), top=4)
