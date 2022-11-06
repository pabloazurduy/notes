## The Gang of Four 
Extracted the [python-patterns guide][1]

> In Python as in other programming languages, this grand principle encourages software architects to escape from Object Orientation and enjoy the simpler practices of Object Based programming instead.

In other words: 

> Favor object composition over class inheritance.

The idea is to compose different classes, so you can avoid the ["subclass explosion"][2] that is a consequence of the inheritance design pattern, were the classes are specialized along the way increasing the number of classes to handle

> inheritance design

```python 
class Figure:
    pass 

class Triangle(Figure):
    pass
    #some methods 

class Rectangle(Figure):
    pass 
    #some other methods 
```

> Composition design 

```python 
class Figure:
    type: Union[Rectangle, Triangle]
```



[comment]: References 
[1]: <https://python-patterns.guide/gang-of-four/composition-over-inheritance/>
[2]: <https://python-patterns.guide/gang-of-four/composition-over-inheritance/#problem-the-subclass-explosion>