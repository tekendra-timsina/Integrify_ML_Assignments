## Exercises from 5th July, deadline 8th July 3 PM. 
## Working format : In new pairs, given by Kedar on July 5th


## 2 Best solutions (descriptions also evaluated) will get extra points to the final exam

## Task 1 (2p)

Write functional programming code to apply a list of functions on a list of integers. Explain how it works as well and how it relates to functional programing.

from functools import reduce
import numpy as np

```sh

def func1(a):
    return a **2 

def func2(a):
    return np.sqrt(a)

x = [func1, func2]
y = [1,2,3,4]
result = [reduce(lambda v, f: f(v), x, val) for val in y]
print(result)
```
>In the above code, we first created two functions and pass it to reduce function along with the seqwuence. Also, the mutable list is passed to the reduce function, where it is in immutable state. 

## Task 2 (4 p)

Find the greatest common divisor of a list of numbers using Reduce. Explain how the code works as well. Explain how it works as well and how it relates to functional programing.
```sh
from math import gcd 
from functools import reduce

num = [500,400,300]
result = reduce(gcd,num)
print(result)
```
>We import gcd function from math module and implement it along with the list of integers in the reduce function imported from functools module. 

## Task 3 (6p)

1. Write a function groupby_demonstrator that takes as a list of tuples as an input (arg data) as well as boolean argument (verify_sorted). If verify_sorted is true, the list is sorted by the first key (0-th tuple element), otherwise it is not sorted.

The function should print out key-value pairs by groups (as output from Python groupby). The print statements between groups should have an empty line. Print statements inside-group should not have empty lines in between (just new lines between records).

```sh
from itertools import groupby

def groupby_demonstrator(data, verify_sorted):
    if verify_sorted:
        sorted_data = sorted(data, key = lambda x:str(x[0]))
        grouped = groupby(sorted_data, key = lambda x:x[0])

        for key, group in grouped:
            print('{}:{}'.format(key, list(group)))
            print('')
    
data = [("University", "Aalto","Helsinki", 5, "Hanken"),('one', 1,'two', 'three', 4), ("Bhuwan", "Jyoti", 2, 1, "Python ")]
result = groupby_demonstrator(data, True)
print(result)
```

2. Add a decorator ‘ensure_sorted_grouper’ that overrides the grouping, by making sure that the list is sorted when an argument ‘verify_sorted’ = True is passed. Otherwise, “You didn’t enforce the order” is printed to the console.


```sh
from itertools import  groupby 

def ensure_sorted_grouper(fcn):
    
    def inner(*args, **kwargs):
        if kwargs.get('verify_sorted') == True:
            sorted_data = sorted(*args, key = lambda x:str(x[0]))
            fcn(sorted_data, **kwargs)
        else:
            print("You didn’t enforce the order")
    return inner

@ensure_sorted_grouper
def groupby_demonstrator(data, **verify_sorted):
    if verify_sorted:
        sorted_data = sorted(data, key = lambda x:str(x[0]))
        grouped = groupby(sorted_data, key = lambda x:x[0])

        for key, group in grouped:
            print('{}:{}'.format(key, list(group)))
            print('')
    
data = [("University", "Aalto","Helsinki", 5, "Hanken"),('one', 1,'two', 'three', 4), ("Bhuwan", "Jyoti", 2, 1, "Python ")]
result = groupby_demonstrator(data, verify_sorted = False)
print(result)
```
## Task 4 (5p)

In your own words, describe what is functional programming and how you can write functional programming code in Python (start here: https://docs.python.org/3/library/functional.html)

>The pure functions that eliminate the side effects of the code upto the extent possible without sacrificing development implementation time are termed as functional programming (FP). Functions in the FP are termed as first class objects and are always higher ordered functions, which means that you should be able to apply all the constructs of using data, to functions as well.THe FP limits the use of for loops, iunstead use recursion.

>In python there are builtin functional programming module that you can implement directly into your higher order functional code. itertools, functools, and operator are the functional programming module in python. 