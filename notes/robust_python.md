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

## Chapter 6 -  Typing



[Comment]: References 
[1]: (https://www.amazon.com/Robust-Python-Write-Clean-Maintainable/dp/1098100662)
