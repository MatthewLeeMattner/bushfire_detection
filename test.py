import numpy as np




arr = np.array([
    [5, 7],
    [0.2, 0.222],
    [0.9, 0.2]
])

print(softmax(arr, axis=1))
'''
from read_data import slice_image

arr = np.array(range(15))
arr = arr[2:7]
print(arr)




arr = np.arange(1, 26)
print(arr.shape)
arr = arr.reshape((5, 5, 1))
print(arr.shape)

result = slice_image(arr, 5, 5)
print(result.shape)
print(result[0])
print("_____________")
print(result[5])
'''