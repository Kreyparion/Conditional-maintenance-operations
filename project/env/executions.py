from typing import Callable, Dict, Union, List, Tuple

from project.env.wearing_functions import wearing_function


def execution(cmd:str)-> Tuple[bool,List[float],List[float],Callable]:
    
    # customisable execution command : "4dsimple" -> 4 items discrete 
    # "12cadvanced" -> 12 items continuous (advanced config)
    nb = ""
    i = 0
    while i<len(cmd) and cmd[i].isdigit():
        nb = nb + cmd[i]
        i += 1
    if nb == "":
        raise NameError("not starting with an integer")
    nb_items = int(nb)
    if not cmd[i] in ["c","d"]:
        raise NameError("should be either (c)ontinuous or (d)iscrete")
    continuous = (cmd[i] == "c")
    
    if not continuous:
        remaining = cmd[i+1:]
        if "advanced" in remaining:
            threshold,wearing_fun = wearing_function("discrete2")
            max_prod = 1.
            return (False ,nb_items * [max_prod],nb_items * [threshold],wearing_fun)
        else:
            threshold,wearing_fun = wearing_function("discrete")
            max_prod = 1.
            return (False ,nb_items * [max_prod],nb_items * [threshold],wearing_fun)
    else:
        remaining = cmd[i+1:]
        if "advanced" in remaining:
            threshold,wearing_fun = wearing_function("continuous2")
            max_prod = 1.
            return (True ,nb_items * [max_prod],nb_items * [threshold],wearing_fun)
        else:
            threshold,wearing_fun = wearing_function("continuous")
            max_prod = 1.
            return (True ,nb_items * [max_prod],nb_items * [threshold],wearing_fun)

