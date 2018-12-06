def getKey(a):
    with open(a, 'r') as f:
        return f.readline().rstrip()
