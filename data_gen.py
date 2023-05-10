import random

fn="data1.txt"

su = 0
to = 0

with open(fn, "w") as file:
    for i in range(10000):
        a = random.choice(range(10))
        b = random.choice(range(10))
        c = random.choice(range(10))
        d = random.choice(range(10))
        e = random.choice(range(10))


        if((a > b  or a < c - d) and b - c < d and (d != e or e < a + b)):
            fin = True
            su += 1

        else:
            fin = False

        to += 1

        msg = str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(fin) + "\n"
        file.write(msg)


print(su/to)


        




