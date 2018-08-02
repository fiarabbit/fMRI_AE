class MyClass:
    def __getattr__(self, item):
        return "undefined"


class_inst = MyClass()
print(class_inst.param)  # "undefined"
print(hasattr(class_inst, "param"))  # True <- OMG!
class_inst.param = "defined"
print(class_inst.param)  # "defined"
print(class_inst.__getattr__("param"))  # "undefined" <- OMG!
print(getattr(class_inst, "param"))  # "defined"
print(hasattr(class_inst, "param"))  # True

