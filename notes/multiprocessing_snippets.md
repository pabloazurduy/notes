*created on: 2022-12-13 11:20:09*
## Multiprocessing Snippets

### run a function with arguments (return in dictionary)
This is a regular mp run with a dictionary

```python 
import multiprocessing as mp
import time 

def my_function(df):
    # do something time consuming
    time.sleep(50)

if __name__ == '__main__':
    with mp.Pool(10) as pool:
        res = {}
        output = {}
        for id, df in some_dict_of_dfs:
            res[id] = pool.apply_async(my_function,(df, ))
        output = {id : res[id].get() for id in id_list}
```


### Run a void process 
this is a snippet to run a function where a complex object is shared. Using [map][1]

```python 
import multiprocess as mp

LIST_OBJ:List[MyClass] = [] # shared global object list 

def my_process(id):
    object = LIST_OBJ[id] 
    object.run_complex_method()
    return object # this is a solved object 

if __name__ == '__main__':
    for i in long_list_df:
        #do something 
        # obj 
        LIST_OBJ.append(obj) 
    
    with mp.Pool(processes=8) as pool:
        result_list = pool.map(f, range(len(LIST_OBJ)))

```

###  Using TQDM progressbar 
Another example but using [tqdm][2] contrib lib [>ref][3] This will display a progress bar when running mp

```python 
from tqdm.contrib.concurrent import process_map 

LIST_OBJ:List[MyClass] = [] # shared global object list 

def my_process(id):
    object = LIST_OBJ[id] 
    object.run_complex_method()
    return object # this is a solved object 

if __name__ == '__main__':
    for i in long_list_df:
        # do something 
        # generate obj 
        LIST_OBJ.append(obj) 
    
    
    result_list = process_map(f, range(len(LIST_OBJ)), max_workers=4)

```




[//]: <> (References)
[1]: <https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers>
[2]: <https://stackoverflow.com/a/59905309/5318634>
[3]: <https://tqdm.github.io/docs/contrib.concurrent/#process_map>

[//]: <> (Some snippets)
[//]: # (add an image <img src="" style='height:400px;'>)