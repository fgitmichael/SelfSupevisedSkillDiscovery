import gtimer as gt

gt.stamp('test_stamp1', unique=True)
for i in gt.timed_for(range(3)):
    a = sum([i for i in range(1000)])
    gt.stamp('test_stamp2', unique=False)

gt.stamp('test_stamp3', unique=False)

a = sum([i for i in range(1000)])

gt.stamp('test_stamp3', unique=False)

stamps = gt.get_times().stamps.itrs
print(stamps)


