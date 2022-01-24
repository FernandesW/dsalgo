from math import pi
import time
from functools import wraps,update_wrapper
class Shape:
    def __init__(self, name):
        self.name = name

    def area(self):
        pass

    def fact(self):
        return "I am a two-dimensional shape."

    def __str__(self):
        return self.name


class Square(Shape):
    def __init__(self, length):
        super().__init__("Square")
        self.length = length

    def area(self):
        return self.length ** 2

    def fact(self):
        return "Squares have each angle equal to 90 degrees."


class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2


def method_overriding():
    a = Square(4)
    b = Circle(7)
    print(b)
    print(b.fact())
    print(a.fact())
    print(b.area())


def decorator_function(original_function):
    @wraps(original_function)
    def wrapper_func(*args, **kwargs):
        print(f'{original_function.__name__} before')
        return original_function(*args, **kwargs)
    return wrapper_func


def checkorator_function(original_function):
    @wraps(original_function)
    def checkorator_func(*args, **kwargs):
        print(f'{original_function.__name__} new before')
        return original_function(*args, **kwargs)
    return checkorator_func


class ClassDecor:
    def __init__(self, original_function):
        update_wrapper(self,original_function)
        self.original_function = original_function

    def __call__(self, *args, **kwargs):
        print('ClassDecor executed ')
        print(f'{self.original_function.__name__} class before')
        return self.original_function(*args, **kwargs)


class CheckClassDecor:
    def __init__(self, original_function):
        update_wrapper(self,original_function)
        self.original_function = original_function

    def __call__(self, *args, **kwargs):
        print('CheckClassDecor executed ')
        print(f'{self.original_function.__name__} class before')
        return self.original_function(*args, **kwargs)


@decorator_function
def display_ini():
    print("display_ini ran")

#
# @decorator_function
# @checkorator_function
@CheckClassDecor
@ClassDecor
def display_names(name, age):
    time.sleep(3)
    print("display_names ran Name:", name, "Age:", age)


if __name__ == '__main__':
    # Method Overriding
    # method_overriding()
    # Decorators
    display_names("John",1)