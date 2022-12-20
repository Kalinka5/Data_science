import numpy as np
from math import log

'''
All Data Type

'i' - integer
'b' - boolean
'u' - unsigned integer
'f' - float
'c' - complex float
'm' - timedelta
'M' - datetime
'O' - object
'S' - string
'U' - unicode string
'V' - fixed chunk of memory for other type ( void )
'''

# create array
x = np.array(range(10), float)  # dtype='f'
# create an array in the interval range
x_2 = np.arange(2, 10, 3)  # = [2, 5, 8]
# create two-dimensional array
x_3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# create 3-dimensional array
x_4 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
# create n-dimensional array
x_5 = np.array([1, 2, 3, 4], ndmin=5)  # = [[[[[1 2 3 4]]]]]

# converting from float type to integer type
int_x = x.astype('i')
int_x_2 = x.astype(int)

# get amount of array dimensions
print(x.ndim)  # = 1
print(x_3.ndim)  # = 2
print(x_4.ndim)  # = 3
print(x_5.ndim)  # = 5

# get amount of elements
print(x.size)  # = 10
print(x_3.size)  # = 9

# get the amount of rows and columns in the matrix
print(x.shape)  # = (10,)
print(x_3.shape)  # = (3, 3)
print(x_4.shape)  # = (2, 2, 3)
print(x_5.shape)  # = (1, 1, 1, 1, 4)

# Mean value
print(x.dtype)  # = float64
print(x_3.dtype)  # = int32

# get length of the first dimension (axis)
print(len(x))  # = 10
print(len(x_3))  # = 3

# array reshaping
reshape_x = x.reshape((5, 2))
print(x)  # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
print(reshape_x)  # [[0. 1.]
#                   [2. 3.]
#                   [4. 5.]
#                   [6. 7.]
#                   [8. 9.]]

reshape_x_3 = x_3.reshape(-1)  # or x_2.flatten()
print(x_3)  # [[1 2 3]
#             [4 5 6]
#             [7 8 9]]
print(reshape_x_3)  # [1 2 3 4 5 6 7 8 9]

# make copy of array
copy_x = x.copy()  # NOT be affected by the changes made to the original array.
# make view of array
view_x = x.view()  # BE affected by the changes made to the original array.

# converting array into list
list_x = x.tolist()
list_x_2 = list(x_3)

# converting array into binary string
binary_bytes_x = x.tobytes()
# converting from binary string to array
x = np.frombuffer(binary_bytes_x)

# Slices
print(x_3[2, :])  # = [7 8 9]
print(x_3[:, 2])  # = [3 6 9]

# check element IN array
print(3 in x)  # = True
print(12 in x)  # = False

# add elements to array
add_10 = np.append(x, 10)

# delete element by index
delete_0 = np.delete(x, 0)

# sorting the array
sort_x = np.sort(x)

# Filtering Arrays
filter_x = x % 2 == 0  # [ True False  True False  True False  True False  True False]
even_x = x[filter_x]  # [0. 2. 4. 6. 8.]

# conditionals (and, or)
print(x[(x > 3) & (x % 2 == 0)])  # = [4. 6. 8.]
print(x[(x > 6) | (x % 2 == 0)])  # = [0. 2. 4. 6. 7. 8. 9.]
print(x[x > 12])  # = []

# operation with array
double_x = x*2
print(double_x)  # = [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18.]

# iterating on each scalar element
for n in np.nditer(x):
    print(n)
# iterating array with different data types
for n in np.nditer(x, flags=['buffered'], op_dtypes=['S']):
    print(n)
# enumerated iteration using element and index of element
for idx, n in np.ndenumerate(x):
    print(idx, n)

# Concatenation
# 1D
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))  # [1, 2, 3, 4, 5, 6]
# 2D
arr1_2D = np.array([[1, 2], [3, 4]])
arr2_2D = np.array([[5, 6], [7, 8]])
arr_2D = np.concatenate((arr1_2D, arr2_2D), axis=1)  # [[1 2 5 6]
#                                                       [3 4 7 8]]

# Stacking function (concatenation)
st_arr1 = np.array([1, 2, 3])
st_arr2 = np.array([4, 5, 6])

# Stacking Along Rows
arr_rows = np.hstack((st_arr1, st_arr2))  # [1 2 3 4 5 6]
# Stacking Along Columns
arr_columns = np.vstack((st_arr1, st_arr2))          # [[1 2 3]
arr_columns_1 = np.stack((st_arr1, st_arr2), axis=0)  # [4 5 6]]
# Stacking Along Height (depth)
height_arr = np.dstack((st_arr1, st_arr2))          # [[1 4]
height_arr_1 = np.stack((st_arr1, st_arr2), axis=1)  # [2 5]
#                                                      [3 6]]

# Splitting Arrays
new_x = np.array_split(x, 4)
print(new_x)  # [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7.]), array([8., 9.])]
print(new_x[0])  # [0. 1. 2.]
print(new_x[-1])  # [8. 9.]

# Searching Array`s index
search_arr = np.array([1, 2, 3, 4, 5, 4, 4])
find_4 = np.where(arr == 4)  # (array([3, 5, 6], dtype=int64),)
find_even_x = np.where(x % 2 == 0)  # (array([0, 2, 4, 6, 8], dtype=int64),)

# sum of array’s elements
sum_x = x.sum()

# minimum and maximum elements in the array
min_x = x.min()
max_x = x.max()

# Mean
mean_x = np.mean(x)
print(mean_x)  # = 4.5

# Median
median_x = np.median(x)
print(median_x)  # = 4.5

# Variance
var_x = np.var(x)
print(var_x)  # 8.25

# Standard deviation
std_x = np.std(x)
print(f"{std_x:.2f}")  # = 2.87

# Ufunc
'''
Create Your Own ufunc

def my_add(num1, num2):
    return num1+num2

ufunc_my_add = np.frompyfunc(my_add, 2, 1)
print(ufunc_my_add([1, 2, 3, 4], [5, 6, 7, 8]))
'''
arr_1 = [10, 21, 38, 4]
arr_2 = [2, 3, 5, 6]

# Add (+)
z = np.add(arr_1, arr_2)  # [5 7 9 11]

# Summations
summa = np.sum([arr_1, arr_2])  # = 89
# Summation Over an Axis
axis_sum = np.sum([arr_1, arr_2], axis=1)  # [73, 16]
# Cumulative Sum
cum_sum = np.cumsum(arr_2)  # = [2, 2+3, 2+3+5, 2+3+5+6] = [2, 5, 10, 16]

# Subtraction (-)
sub = np.subtract(arr_1, arr_2)  # [8, 18, 33, -2]

# Multiplication (*)
mult = np.multiply(arr_1, arr_2)  # [20, 63, 190, 24]

# Division (/)  # [5, 7, 12.6666, 0.6666]
div = np.divide(arr_1, arr_2)

# Power (**) the first array to the power of the values of the second array
power = np.power(arr_1, arr_2)

# Remainder (%) mod and remainder the same
rem_1 = np.mod(arr_1, arr_2)
rem_2 = np.remainder(arr_1, arr_2)

# Quotient and Mod (// and %)
quot_and_mod = np.divmod(arr_1, arr_2)  # (array([5, 7, 7, 0]), array([0, 0, 3, 4]))

# Products
prod = np.prod(arr_2)  # = 2*3*5*6 = 180
prod_2 = np.prod([arr_1, arr_2])  # = 10*21*38*4*2*3*5*6 = 5745600
# Product Over an Axis
prod_3 = np.prod([arr_1, arr_2], axis=1)  # [31920, 180]
# Cumulative Product
prod_4 = np.cumprod(arr_2)  # = [2, 2*3, 2*3*5, 2*3*5*6] = [2, 6, 30, 180]

# Differences
dif = np.array([10, 15, 25, 5])
new_dif = np.diff(arr)  # [15-10, 25-15, 5-10] = [5, 10, -20]
# Compute discrete difference of the following array n-times:
new_dif_2 = np.diff(arr, n=2)  # [15-10, 25-15, 5-10] = [5, 10, -20] = [10-5, -20-10] = [5, -30]

# Absolute Values
some_arr = np.array([-1, -2, 1, 2, 3, -4])
abs_arr = np.absolute(arr)  # [1, 2, 1, 2, 3, 4]

# Rounding Decimals
r_arr = np.around(3.1666, 2)  # where second number is decimal places
fix = np.trunc(3.6666)
print(fix)  # = 3.17

# Logarithm
arr = np.arange(1, 10)
# Log at Base 2
print(np.log2(arr))
# Log at Base 10
print(np.log10(arr))
# Natural Log, or Log at Base e
print(np.log(arr))
# Log at Any Base. We need from math import log.
np_log = np.frompyfunc(log, 2, 1)
print(np_log(100, 15))  # = 1.7005483074552052

# Finding LCM (Lowest Common Multiple)
num1 = 14
num2 = 6
lcm = np.lcm(num1, num2)  # = 42
# Finding LCM in Arrays
lcm_array = np.array([3, 7, 15])
lcm_2 = np.lcm.reduce(lcm_array)  # = 105

# Finding GCD (Greatest Common Denominator)
gcd = np.gcd(num1, num2)  # = 2
# Finding GCD in Arrays
lcm_array = np.array([4, 432, 40])
gcd_2 = np.gcd.reduce(lcm_array)  # = 4
