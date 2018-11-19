import json
class ConfigWrapper(object):
    def __init__(self, attrs):
        if 'self' in attrs:
            attrs['self'] = attrs['self'].__class__.__name__
        config = dict()
        for key, val in attrs.items():
            self.__setattr__(key, val)
            config[key] = str(val)

        self.config = config
