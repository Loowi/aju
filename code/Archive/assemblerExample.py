import dis

def while_one():
    while 1:
        pass

def while_true():
    while True:
        pass

print("while 1")
print("----------------------------")
dis.dis(while_one)

print("while True")
print("----------------------------")
dis.dis(while_true)