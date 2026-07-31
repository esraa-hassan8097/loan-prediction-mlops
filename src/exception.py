import sys
from src.logger import logging

#
class CustomException(Exception):#we are saying here Create a new type of exception based on Python's standard exception."This is called inheritance.
    #Python already has a built-in class called Exception, so we are inheriting from it to create our own custom exception class.
    #pass #Create the class, but don't give it any custom behavior yet.


       def __init__(self, error_message, error_detail: sys): #init function runs automatically when we create an object.
        self.error_message = error_message #Now every CustomException object remembers its own error message.??

        _, _, exc_tb = error_detail.exc_info() #means:Ignore exception type Ignore exception object Keep traceback