
class GUIParam():
    """
    Base class for GUI params. Brings together FAST parameter and GUI elements 
    so that they stay in sync. 

    Parameters:
        param (FASTParam): FAST config parameter object 
    """
    def __init__(self, param):

        self.elements = []
        self.param = param 

    