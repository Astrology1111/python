def get_my_counter():
    x = -1
    def my_counter():
        nonlocal x
        x += 1
        return x
    return my_counter