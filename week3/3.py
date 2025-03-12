def my_sum(*args,value = 1):
    '''Add Value To Numbers'''
    result = []
    for i in range(len(args)):
        result.append(value + args[i])
    return result
