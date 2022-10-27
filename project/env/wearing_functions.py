from typing import Callable, Tuple 


def wearing_function(function_type:str) -> Tuple[float,Callable]:
    linked_functions = {
        "discrete": discrete_wearing_function(),
        "discrete2": discrete_wearing_function2(),
        "continuous": continuous_wearing_function(),
        "continuous2": continuous_wearing_function2(),
    }
    if not function_type in linked_functions.keys():
        raise NameError("wrong function name")
    return linked_functions[function_type]



def discrete_wearing_function()-> Tuple[float,Callable]:
    """ return threshold and function"""
    def fun():
        pass
    threshold = 5
    return threshold,fun

def discrete_wearing_function2()-> Tuple[float,Callable]:
    """ return threshold and function"""
    def fun():
        pass
    threshold = 5
    return threshold,fun

def continuous_wearing_function()-> Tuple[float,Callable]:
    """ return threshold and function"""
    def fun():
        pass
    threshold = 5
    return threshold,fun

def continuous_wearing_function2()-> Tuple[float,Callable]:
    """ return threshold and function"""
    def fun():
        pass
    threshold = 5
    return threshold,fun