
- use **set** to find unique word from a list of words.

my_dict = dict()
my_dict['ank'] = 20
my_dict['ban'] = 30
// sort by values and show result in descending order
sorted(my_dict, reverse=True, key=lambda x: x[1])




'abcdefghijklm'[::3]  # beginning to end, counting by 3
'adgjm'

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
