def run_once(f):
        """
        Decorates functions and ensures they're never run more than once. Useful
        for initializations and checking variables because I'm a hack.
        """

        def wrapper(*args, **kwargs):
                if not wrapper.has_run:
                        wrapper.has_run = True
                        return f(*args, **kwargs)
                wrapper.has_run = False
                return wrapper
