from dataclasses import DataClass
from 

from x import Group, Dimension, Variable

@DataClass
class P5Contract:
    """
    This class defines the contract between the local and remote sides of the system.
    """ 
    attrs : dict[str, Any]
    dimensions : dict[str, Dimension]
    groups: dict[str, Group]
    variables: dict[str, Variable]