def generate_fibo(limit):
    """
    Generates a fibonacci sequence that is less than the limit.
    :param limit: upper limit of fibonacci sequence
    :return: next fibonacci sequence entry
    """
    current_num = 0
    next_num = 1
    while current_num < limit:
        yield current_num

        # old_num = current_num
        # current_num = next_num
        # next_num = old_num + current_num

        current_num, next_num = next_num, current_num + next_num

for i in generate_fibo(-1):
    print(i)
