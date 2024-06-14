import fast 
import time 

t0 = time.perf_counter()
sim = fast.Fast("test_params.py")
t1 = time.perf_counter()
sim.run()
t2 = time.perf_counter() 

print("FAST Benchmark using test_params.py:")
print(f"Initialisation time: {t1-t0} s")
print(f"Iteration time: {t2-t1} s")
print(f"Total time: {t2-t0} s")