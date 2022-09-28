class A:
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return self.value**2


class B(A):
    def __init__(self, other):
        # I know this is wrong, but this is sort of what I want to do.
        # In the real use-case, A is enormously complex and big (so an iteration
        # over instance attributes will be messy and had better not involve copies),
        # and the instance is generated deep in a method I can't modify so
        # I can't replace the creation of A with a creation of B.
        self = other
    
    def process(self):
        return self.value**3

def process(self):
    return self.value**3


def make_an_a_instance_effectively_a_b(an_instance):
    """ This is a hack to get around the problem with initialising an instance of the B class """
    # see https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
    an_instance.process = process.__get__(an_instance, A)
    return an_instance


if __name__=="__main__":
    a = A(5)
    print(a.process())

    b = make_an_a_instance_effectively_a_b(a)
    print(b.process())

    b = B(a)
    print(b.process())