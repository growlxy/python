import time
'''
for i in range(1, 5):
    print(f'\r{i}%', end='')
    time.sleep(1)
'''
scale=50
print("执行开始".center(scale//2,"-"))
start=time.perf_counter()
for i in range(scale+1):
    fx=int(pow((i+(1-i)*0.03),2))
    a="*"*i
    b='-'*(scale-i)
    c=(fx/pow((50+(1-50)*0.03),2))*100
    dur=time.perf_counter()-start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur),end="")
    time.sleep(0.5)
