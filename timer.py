def timer(now_time):
    def deco(func):
        def wrapper(*args, **kwargs):
            t1 = time.perf_counter()
            print(f'现在是{now_time}')

            func(*args, **kwargs)
            t2 = time.perf_counter()
            print(f'{round(t2 - t1, 4)}s\n')
        return wrapper
    return deco
