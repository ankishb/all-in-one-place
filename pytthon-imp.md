
## Access info, what is inside the class **Use help, dir, vars**
```python
[x for x in dir(class_object) if not x.startswith("__")]
```


## use **set** to find unique word from a list of words.

```python
my_dict = dict()
my_dict['ank'] = 20
my_dict['ban'] = 30
// sort by values and show result in descending order
sorted(my_dict, reverse=True, key=lambda x: x[1])
```


## slicing

```python
'abcdefghijklm'[::3]  # beginning to end, counting by 3
'adgjm'
```

## Reduce workload while using function with many many parameters, of which few are fixed

```python
def power(base, exponent):
    return base ** exponent

def square(base):
    return power(base, 2)

def cube(base):
    return power(base, 3)


from functools import partial

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

def test_partials():
    assert square(2) == 4
    assert cube(2) == 8
```

## slicing in much efficient way
 
For example, you can now easily extract the elements of a list that have even indexes:

```python
>>> L = range(10)
>>> L[::2]
[0, 2, 4, 6, 8]
```
Negative values also work to make a copy of the same list in reverse order:
```python
>>> L[::-1]
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```
