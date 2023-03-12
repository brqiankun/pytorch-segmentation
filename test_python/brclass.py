class A:
    def add(self, x):
        y = x + 1
        print(y)

class B(A):
    def add(self, x):
        super().add(x)

b = B()
b.add(2)

class FooParent(object):  #object是
    def __init__(self) -> None:
        self.parent = 'I\'m the parent.'
        print('Parent')
        
    def bar(self, message):
        print("%s from parent" %message)

class FooChild(FooParent):
    def __init__(self) -> None:
        super(FooChild, self).__init__()
        print("Child")
        
    def bar(self, message):
        super(FooChild, self).bar(message)
        print('child bar function')
        print(self.parent)
        

# namespace test  scope 作用域
def scope_test():
    def do_local():
        spam = "local spam"
    
    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"
    
    def do_global():
        global spam
        spam = "global spam"
    
    spam = "test spam"
    do_local()
    print("after local assignment:", spam)
    do_nonlocal()
    print("after nonlocal assignment:", spam)
    do_global()
    print("after global assignment:", spam)
    

class MyClass:
    """A simple example class"""
    i = 12345
    
    # 新创建实例时，自动调用__init__() 方法
    def __init__(self) -> None:  
        self.data = []
    
    def f(self):
        return 'hello world'
    

x = MyClass()

class Complex:
    def __init__(self, realpart, imagpart) -> None:
        self.r = realpart
        self.i = imagpart

class Dog:
    kind = 'canine'
    
    def __init__(self, name) -> None:
        self.name = name
        
d = Dog('Fido')
e = Dog('Buddy')

print(d.kind)
print(e.kind)
print(d.name)
print(e.name)



    
      
        
        
        
if __name__ == "__main__":
    # fooChild = FooChild()
    # fooChild.bar("hello world")
    scope_test()
    print("in global scope:", spam)
    
    