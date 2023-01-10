import sys
import numpy as np

REFERENCE = False
STEP = -1
steps = {}


def new_step():
    global STEP
    STEP += 1
    steps[STEP] = {"base_cycles": {}, "events": {}}


def get_device_log(array):
    global REFERENCE, STEP, steps
    event = int.from_bytes(array[:4], byteorder='little')
    funcID = int.from_bytes(array[4:8], byteorder='little')
    numInvocations = int.from_bytes(array[8:12], byteorder='little')
    nodeID = int.from_bytes(array[12:16], byteorder='little')
    blockID = int.from_bytes(array[16:20], byteorder='little')
    smID = int.from_bytes(array[20:24], byteorder='little')
    cycleCount = int.from_bytes(array[24:32], byteorder='little')
    if event == 0:
        assert funcID == 0 and numInvocations == 0 and nodeID == 0
        if not REFERENCE:
            new_step()
            REFERENCE = True
        if smID not in steps[STEP]["base_cycles"]:
            steps[STEP]["base_cycles"][smID] = {blockID: cycleCount}
        else:
            assert blockID not in steps[STEP]["base_cycles"][smID]
            steps[STEP]["base_cycles"][smID][blockID] = cycleCount
    else:
        REFERENCE = False
        if nodeID not in steps[STEP]["events"]:
            steps[STEP]["events"][nodeID] = {
                "func": (funcID, numInvocations),
                event: (smID, blockID, cycleCount)
            }
        else:
            assert event not in steps[STEP]["events"][nodeID]
            steps[STEP]["events"][nodeID][event] = (smID, blockID, cycleCount)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python parse_tracing.py [file_name]")
        exit()

    with open(sys.argv[1], 'rb') as f:
        events = bytearray(f.read())
        assert len(events) % 32 == 0
        print("# logs", len(events) / 32)
        for i in range(0, len(events), 32):
            get_device_log(events[i:i + 32])

    e = steps[8]

    variance = [
        max(v.values()) - min(v.values()) for v in e["base_cycles"].values()
    ]
    sm_base_average = {
        k: sum(v.values()) / len(v.values())
        for k, v in e["base_cycles"].items()
    }
    print("base cycle variance on each sm:", variance)

    def calibrate(timestamp):
        sm, block, cycle = timestamp
        return cycle - sm_base_average[sm]

    duration = {}
    for i in range(1, max(e["events"])):
        if i not in e["events"]:
            continue
        duration[i] = (*e["events"][i]["func"], calibrate(e["events"][i][2]) -
                       calibrate(e["events"][i][1]))

    # print(duration)

    sorted_duration = dict(
        sorted(duration.items(), key=lambda item: item[1][2]))

    total = sum(v[2] for v in sorted_duration.values())
    normailized = {
        k: (v[0], v[1], round(v[2] / total, 3))
        for k, v in sorted_duration.items()
    }
    print(normailized)

    func_percentage = {}
    for k, v in normailized.items():
        if v[0] not in func_percentage:
            func_percentage[v[0]] = [{k: v[1]}, v[2]]
        else:
            assert k not in func_percentage[v[0]][0]
            func_percentage[v[0]][0][k] = v[1]
            func_percentage[v[0]][1] += v[2]
    sorted_func = dict(
        sorted(func_percentage.items(), key=lambda item: item[1][1]))
    print(sorted_func)
