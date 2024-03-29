Notes from the book [Robust Python][1]

## Chapter 4 -  Typing

Some useful declarations 

- Literal

    Set of alternatives 
    ```python
    Literal['one','twos']
    ```

- NewType 

    build a custom type to differentiate a changed object without access to his inner state (is this a good pattern ?), also this typing is not an alias because it defines a unique type 
    ```python 
    class Figure:
        pass
    FigureReshaped = NewType('FigureReshaped', Figure)
    ```
- Type Alias 

    you can build an alias directly declaring them 
    ```python 
    IdOrName = Union[str,int]
    ```
- Final
    
    a type for constants, it raise typing errors when reassigned, no when mutate

    ```python
    SOME_CONSTANT:Final = 3
    SOME_CONSTANT = 5 # this throws an error (reassign)
    # we can mutate it f(x) = x+1 : f(SOME_CONSTANT) no error  
    ```

## Chapter 8 -  Enums

```python
from enum import Enum, auto
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = auto() # you can change the values to auto if you don't care about the values
```
`Literal` and `Enum` had a similar purpose, you usually use `Enums` when you also need to iterate over the values, otherwise `Literal` is enough for type checking

`Flag`'s are another class of enum where you can combine them togetuer using `&, |, ^, ~`  and the result will also be a flag from the same class 

```python
from enum import Flag, auto
class Color(Flag):
    RED = auto() # use auto! (a defined number or string will generate problems)
    BLUE = auto()
    GREEN = auto()
    WHITE = Color.RED & Color.GREEN & Color.BLUE #its a color too 
    PRIMARY = Color.RED | Color.GREEN | Color.BLUE # is on if any other is on too 
```

`IntEnum` and `IntFlag` are the same as `Enum` and `Flag` but the value of the elements is forced to be an integer, this is good for comparison enums

```python
from enum import IntEnum
class Priority(IntEnum):
    HIGH = 5
    MID = 1 
    LOW = 0

print(Priority.HIGH > Priority.LOW) # >> True
```

## Chapter 9 - DataClasses 

Equality `eq=True`, using `@dataclass(eq=True)` option will automatically create a `__eq__()` method that will compare all the fields in a class against another object from the same class to verify if they are equal. 

using `frozen=True` will only prevent to reasign the pointers from the attributes from a class, but you will be still able to mutate the objects on the pointers. 
    
```python 
    object.attr = 0 # this will throw an error
    object.attr.some_subattr = 0 # this is allowed
```
## Chapter 10 - Classes 
**SOLID** principles:
1. S "Single responsibility Principle", any class should only have **one responsibility** -any object should have only one reason "to change"-


## Chapter 12 - "Subtyping"

The Liskov Substitution Principle (LSP): It states that objects of a `superclass` should be replaceable with objects of its `subclasses` (inheritance) without affecting the correctness of the program. In other words, if a program is designed to work with a certain type of object, it should also work correctly with any subtype of that object. Violating the LSP can lead to unexpected behavior and can make the codebase more fragile.


[Comment]: References 
[1]: <https://www.amazon.com/Robust-Python-Write-Clean-Maintainable/dp/1098100662>
