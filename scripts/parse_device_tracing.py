import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

REFERENCE = False
STEP = -1
LOG_STEPS = {}


def new_step():
    global STEP
    STEP += 1
    LOG_STEPS[STEP] = {
        "base_cycles": {},
        "events": {},
        "SMs": {},
        "mapping": {}
    }


def get_device_log(array):
    global REFERENCE, STEP, LOG_STEPS
    event = int.from_bytes(array[:4], byteorder='little')
    funcID = int.from_bytes(array[4:8], byteorder='little')
    numInvocations = int.from_bytes(array[8:12], byteorder='little')
    nodeID = int.from_bytes(array[12:16], byteorder='little')
    blockID = int.from_bytes(array[16:20], byteorder='little')
    smID = int.from_bytes(array[20:24], byteorder='little')
    cycleCount = int.from_bytes(array[24:32], byteorder='little')

    if event in [1, 2]:
        if nodeID not in LOG_STEPS[STEP]["mapping"]:
            LOG_STEPS[STEP]["mapping"][nodeID] = (funcID, numInvocations)
        else:
            assert LOG_STEPS[STEP]["mapping"][nodeID] == (funcID,
                                                          numInvocations)

    if event == 0:
        assert funcID == 0 and numInvocations == 0 and nodeID == 0
        if not REFERENCE:
            new_step()
            REFERENCE = True
        if smID not in LOG_STEPS[STEP]["base_cycles"]:
            LOG_STEPS[STEP]["base_cycles"][smID] = {blockID: cycleCount}
        else:
            assert blockID not in LOG_STEPS[STEP]["base_cycles"][smID]
            LOG_STEPS[STEP]["base_cycles"][smID][blockID] = cycleCount
    else:
        REFERENCE = False

        if event in [3, 4]:
            if smID not in LOG_STEPS[STEP]["SMs"]:
                assert event == 3
                LOG_STEPS[STEP]["SMs"][smID] = {
                    (numInvocations, nodeID, blockID): [cycleCount]
                }
            else:
                if (numInvocations, nodeID,
                        blockID) in LOG_STEPS[STEP]["SMs"][smID]:
                    assert event == 4
                    LOG_STEPS[STEP]["SMs"][smID][(numInvocations, nodeID,
                                                  blockID)].append(cycleCount)
                else:
                    LOG_STEPS[STEP]["SMs"][smID][(numInvocations, nodeID,
                                                  blockID)] = [cycleCount]
            return

        if nodeID not in LOG_STEPS[STEP]["events"]:
            LOG_STEPS[STEP]["events"][nodeID] = {
                event: (smID, blockID, cycleCount)
            }
        else:
            assert event not in LOG_STEPS[STEP]["events"][nodeID]
            LOG_STEPS[STEP]["events"][nodeID][event] = (smID, blockID,
                                                        cycleCount)


def calibrate(timestamp, sm_base_avg):
    sm, _, cycle = timestamp
    return cycle - sm_base_avg[sm]


def serialized_analysis(step_log):

    node_exec_duration = {}
    for i in range(max(step_log["events"])):
        # skipped nodes
        if i not in step_log["events"]:
            continue
        node_exec_duration[i] = (
            *step_log["mapping"][i],
            calibrate(step_log["events"][i][2], step_log["sm_base_avg"]) -
            calibrate(step_log["events"][i][1], step_log["sm_base_avg"]))

    sorted_duration = dict(
        sorted(node_exec_duration.items(), key=lambda item: item[1][2]))

    total_exec_time = sum(v[2] for v in sorted_duration.values())
    normailized = {
        k: (v[0], v[1], round(v[2] / total_exec_time, 3))
        for k, v in sorted_duration.items()
    }
    print("execution time percentage for each node", normailized)

    top10_nodes = {
        i: (normailized[i][-1],
            calibrate(step_log["events"][i][1], step_log["sm_base_avg"]),
            calibrate(step_log["events"][i][2], step_log["sm_base_avg"]))
        for i in list(normailized.keys())[-10::]
    }

    print("top 10 nodes amounts",
          sum([i[-1] for i in list(normailized.values())][-10::]),
          "% of execution time")

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
    print("execution time percentage for each func", sorted_func)
    print("total execution time", total_exec_time)

    return top10_nodes


def block_analysis(step_log):
    sm_execution = {k: 0 for k in step_log["SMs"].keys()}
    sm_final_cycles = {k: 0 for k in step_log["SMs"].keys()}
    block_exec_time = {
        "blocks": {k: {}
                   for k in step_log["SMs"].keys()},  # horizontal
        "nodes": {k: {}
                  for k in step_log["SMs"].keys()},  # vertical
    }

    for sm in step_log["SMs"]:
        for (_, nodeID, blockID), [start, end] in step_log["SMs"][sm].items():
            # confirm clock does proceed within an SM
            assert end > start
            assert start > step_log["sm_base_avg"][sm]

            sm_final_cycles[sm] = max(sm_final_cycles[sm], end)

            start, end = [
                i - step_log["sm_base_avg"][sm] for i in [start, end]
            ]
            sm_execution[sm] += end - start

            if blockID not in block_exec_time["blocks"][sm]:
                block_exec_time["blocks"][sm][blockID] = [(start, end, nodeID)]
            else:
                assert start > block_exec_time["blocks"][sm][blockID][-1][1]
                block_exec_time["blocks"][sm][blockID].append(
                    (start, end, nodeID))

            if nodeID not in block_exec_time["nodes"][sm]:
                block_exec_time["nodes"][sm][nodeID] = []
            block_exec_time["nodes"][sm][nodeID].append((start, end, blockID))

        block_exec_time["blocks"][sm] = {
            k: sorted(v)
            for k, v in block_exec_time["blocks"][sm].items()
        }
        block_exec_time["nodes"][sm] = {
            k: sorted(v)
            for k, v in block_exec_time["nodes"][sm].items()
        }

    # total_time = max([
    #     sm_final_cycles[k] - step_log["sm_base_avg"][k]
    #     for k in sm_final_cycles.keys()
    # ])

    step_log["sm_final_cycles"] = {
        k: sm_final_cycles[k] - step_log["sm_base_avg"][k]
        for k in sm_final_cycles.keys()
    }
    total_time = max(list(step_log["sm_final_cycles"].values()))

    step_log["sm_final_max"] = total_time

    sm_lifetime = sorted(
        [sm_execution[k] / total_time for k in sm_final_cycles.keys()])
    print(sm_lifetime)
    print(
        "on average",
        sum(sm_lifetime) / len(sm_lifetime),
        "% of the time the SM is running, note the real utilization could be even lower"
    )

    return block_exec_time


COLORS = ["red", "green", "blue", "yellow", "purple", "cyan", "pink", "orange"]


def draw(step_log, nodes, blocks):
    num_sm = len(blocks)
    num_block_per_sm = 4
    num_pixel_per_sm = (num_block_per_sm + 1) * 2
    x_limit = 4000
    y_limit = num_sm * num_pixel_per_sm

    colors = {}
    for n in nodes:
        func = step_log["mapping"][n]
        if func not in colors:
            colors[func] = COLORS[len(colors)]
    print(colors)

    img = Image.new("RGB", (x_limit, y_limit), "white")
    draw = ImageDraw.Draw(img)

    def cast_coor(timestamp, limit=x_limit):
        assert (timestamp <= step_log["sm_final_max"])
        return int(timestamp / step_log["sm_final_max"] * limit)

    # print(blocks)

    for s, b in blocks.items():
        y = s * num_pixel_per_sm
        for i in range(num_pixel_per_sm):
            draw.line((cast_coor(
                step_log["sm_final_cycles"][s]), y + i, x_limit, y + i),
                      fill="grey",
                      width=1)

        for bb, events in b.items():
            for e in events:
                draw.line((cast_coor(e[0]), y, cast_coor(e[1]), y),
                          fill=colors[step_log["mapping"][e[2]]]
                          if step_log["mapping"][e[2]] in colors else "black",
                          width=2)
            y += 2
    # print(blocks)

    # mark the start and the end of major nodes
    for k, v in nodes.items():
        draw.line((cast_coor(v[1]), 0, cast_coor(v[1]), y_limit),
                  fill="red",
                  width=1)
        draw.line((cast_coor(v[2]), 0, cast_coor(v[2]), y_limit),
                  fill="green",
                  width=1)

    img.save("test.png")


def step_analysis(step=5):
    step_log = LOG_STEPS[step]

    variance = [
        max(v.values()) - min(v.values())
        for v in step_log["base_cycles"].values()
    ]
    print("step #", step, " base cycle variance on each sm:", variance)

    step_log["sm_base_avg"] = {
        k: sum(v.values()) / len(v.values())
        for k, v in step_log["base_cycles"].items()
    }

    # manually add the nodeStart event for node 0
    step_log["events"][0][1] = (0, 0, step_log["sm_base_avg"][0])

    node_exec_time = serialized_analysis(step_log)

    block_exec_time = block_analysis(step_log)
    draw(step_log, node_exec_time, block_exec_time["blocks"])
    exit()

    def calibrate(sm, cycle):
        return cycle - sm_base_avg[sm]

    calibrated_node_block_time = {}
    for sm in block_exec_time:
        calibrated_node_block_time[sm] = {
            k: [(calibrate(sm, vv[0]), calibrate(sm, vv[1])) for vv in v]
            for k, v in block_exec_time["nodes"][sm].items()
        }
    print(calibrated_node_block_time)
    node_sm_percentage = {k: 0 for k in node_exec_time.keys()}
    # for node in node_sm_percentage:
    #     start, end = node_exec_time[node]
    #     for v in calibrated_node_block_time.values():
    #         nodes_time = {k[0]:v for k,v in v.items()}
    #         if node not in nodes_time:
    #             continue

    # print(node_exec_time)


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

    # print(duration)

    step_analysis(5)
    exit()

    # base = sm_base_avg[sm]
    # z_stack = [sm_final_cycles[sm]]
    # for a, z in durations:
    #     depth = len(z_stack) - 1
    #     if depth not in sm_stacks:
    #         sm_stacks[depth] = 0
    #     sm_stacks[depth] += a - base
    #     base = a
    #     z_stack.append(end)
    #     print(z_stack, a)
    #     while a >= min(z_stack):
    #         sm_stacks[depth] -= (a - min(z_stack))
    #         z_stack.remove(min(z_stack))
    #         depth -= 1

    # print(sm_base_avg[sm], durations, sm_final_cycles[sm])
    # print(sm_final_cycles[sm] - sm_base_avg[sm], sm_stacks)

    # print(max(sm_base_avg.values()) - min(sm_base_avg.values()))
    # print(
    #     sorted([
    #         sm_final_cycles[k] - sm_base_avg[k]
    #         for k in sm_base_avg.keys()
    #     ]))
    # print(
    #     sorted([
    #         sm_execution[k] / (sm_final_cycles[k] - sm_base_avg[k])
    #         for k in sm_base_avg.keys()
    #     ]))
