class WrapperBase(object):

    def __init__(self, instance_to_wrap):
        self.wrapped = instance_to_wrap

    def __getattr__(self, item):
        assert item not in dir(self)
        return self.wrapped.__getattribute__(item)