

with open("nohup.out", "r") as f:
    for line in f.readlines():
        if "Average cost on validation set at step" in line:
            text = line.replace("Average cost on validation set at step", "").replace(" ", "").replace("\n", "")
            step = text.split(":")[0]
            count = text.split(":")[1]
            print(step, count, end=" ")

        if "Doc Avg Length at step" in line:
            text = line.replace("Doc Avg Length at step", "").replace("\n", "")
            step = text.split(":")[0]
            count = text.split(":")[1]
            print(step, count, end=" ")

        if "Query Avg Length at step" in line:
            text = line.replace("Query Avg Length at step", "").replace("\n", "")
            step = text.split(":")[0]
            count = text.split(":")[1]
            print(" || ,q:", count)
