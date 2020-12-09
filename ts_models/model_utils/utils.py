def is_listy(x):
    return isinstance(x, (tuple, list))


class Hook:
    def __init__(self, m, hook_func, is_forward, detach):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (
                (o.detach() for o in output) if is_listy(output) else output.detach()
            )
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None
