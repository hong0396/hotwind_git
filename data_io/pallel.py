
# import time
# from multiprocessing import Pool
# import numpy as np
# import math, sys, time

# ncpus = (sys.argv)
# print(ncpus)
# pCOs = np.linspace(1e-5, 0.5, 10)
# if "__main__" == __name__:
#     # try:
#         start = time.time()
#         pool = Pool()                # 创建进程池对象，进程数与multiprocessing.cpu_count()相同
#         tofs = pool.map(abs, pCOs,4)  # 并行执行函数
#         end = time.time()
#         t = end - start
#         print(t)


from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
