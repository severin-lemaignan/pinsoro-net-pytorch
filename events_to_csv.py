#! /usr/bin/env python3
# coding: utf-8
from tensorboard.backend.event_processing import event_accumulator
import sys

ea = event_accumulator.EventAccumulator(sys.argv[1])
ea.Reload()

tags = ea.Tags()["scalars"]

print("%s" % (",".join(["iteration"] + tags)))

data = [ea.Scalars(t) for t in tags]

vals=[]

for i in range(len(data[0])):
    vals.append(str(data[0][i].step))
    for j in range(len(tags)):
        vals.append(str(data[j][i].value))

    print("%s" % ",".join(vals))
    vals=[]
