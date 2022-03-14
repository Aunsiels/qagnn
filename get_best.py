from sys import argv


START = "| epoch  "


ACCS = []
seems_final = False


with open(argv[1]) as f:
    for line in f:
        if line.startswith(START):
            for param in line.split("|"):
                if "dev_acc" in param:
                    dev = float(param.split()[1])
                elif "test_acc" in param:
                    test = float(param.split()[1])
            ACCS.append((dev, test))
        if line.startswith("-"):
            seems_final = True
        else:
            seems_final = False


ACCS = sorted(ACCS, key=lambda x: -x[0])


print("Best dev accuracy:", ACCS[0][0])
print("Best test accuracy:", ACCS[0][1])
if seems_final:
    print("Seems final")
else:
    print("Seems not finished yet")
