hello = True

def fun (x) :
    if hello :
        print(x)
    else:
        print('goodbye')

fun('hello')
hello = False
fun('hello')