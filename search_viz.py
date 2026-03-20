import glob, os

nums = sorted(list(set(
    [
        int(x.split(os.path.sep)[1]) for x in
        glob.glob(os.path.join('experiments', '*', '*.gif'))
    ]
)))

for num in nums:
    print(num)