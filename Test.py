from numpy import True_


with open('TrainData.txt') as f:
    UNlines = f.readlines()

f.close()

DATA = []
X = []
Y = []

Lines = [s.replace("\n","") for s in UNlines]
Lines = [s.replace(" ",",") for s in Lines]

for i in range(len(Lines)):
    li = list(Lines[i].split(","))
    DATA.append(li)

for i in range(len(Lines)):
    tempt = []
    for j in range(len(DATA[i])-1):
        tempt.append(int(DATA[i][j]))
    X.append(tempt)
    
    if DATA[i][-1] == 'True':
        Y.append(1)
    else:
        Y.append(0)



