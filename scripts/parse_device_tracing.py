import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
import os

STEP = -1
NEW_STEP = False
LOG_STEPS = {}
# FUNC_OFFSET_RANKINGS = {}


def new_step():
    global STEP
    STEP += 1
    LOG_STEPS[STEP] = {
        "base_cycles": {},
        "events": {},
        "SMs": {},
        "mapping": {},
        "final_cycles": {},
        "start_timestamp": 0xFFFFFFFFFFFFFFFF,
        "final_timestamp": 0
    }


def get_device_log(array):
    global NEW_STEP, STEP, LOG_STEPS
    event = int.from_bytes(array[:4], byteorder='little')
    funcID = int.from_bytes(array[4:8], byteorder='little')
    numInvocations = int.from_bytes(array[8:12], byteorder='little')
    nodeID = int.from_bytes(array[12:16], byteorder='little')
    blockID = int.from_bytes(array[16:20], byteorder='little')
    smID = int.from_bytes(array[20:24], byteorder='little')
    cycleCount = int.from_bytes(array[24:32], byteorder='little')

    if STEP == -1 and event != 0:
        print("the log file might be corrupted")
        return

    if event in [1, 2]:
        if nodeID not in LOG_STEPS[STEP]["mapping"]:
            LOG_STEPS[STEP]["mapping"][nodeID] = (funcID, numInvocations)
        else:
            assert LOG_STEPS[STEP]["mapping"][nodeID] == (funcID,
                                                          numInvocations)

    if event == 0:
        assert funcID == 0 and numInvocations == 0 and nodeID == 0
        if not NEW_STEP:
            NEW_STEP = True
            new_step()
        if smID not in LOG_STEPS[STEP]["base_cycles"]:
            LOG_STEPS[STEP]["base_cycles"][smID] = {blockID: cycleCount}
        else:
            assert blockID not in LOG_STEPS[STEP]["base_cycles"][smID]
            LOG_STEPS[STEP]["base_cycles"][smID][blockID] = cycleCount

        LOG_STEPS[STEP]["start_timestamp"] = min(
            LOG_STEPS[STEP]["start_timestamp"], cycleCount)
        return

    NEW_STEP = False

    if event in [1, 2]:
        if nodeID not in LOG_STEPS[STEP]["events"]:
            LOG_STEPS[STEP]["events"][nodeID] = {
                event: (smID, blockID, cycleCount)
            }
        else:
            assert event not in LOG_STEPS[STEP]["events"][nodeID]
            LOG_STEPS[STEP]["events"][nodeID][event] = (smID, blockID,
                                                        cycleCount)
        return

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

    if event == 5:
        assert blockID not in LOG_STEPS[STEP]["final_cycles"]
        LOG_STEPS[STEP]["final_cycles"][blockID] = cycleCount
        LOG_STEPS[STEP]["final_timestamp"] = max(
            LOG_STEPS[STEP]["final_timestamp"], cycleCount)


def serialized_analysis(step_log):

    def calibrate(timestamp, sm_base_avg):
        sm, _, cycle = timestamp
        return cycle - sm_base_avg[sm]

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
        for i in list(normailized.keys())[:-11:-1]
    }

    print("top 10 nodes amounts {:.3f}% of execution time".format(
        sum([i[-1] for i in list(normailized.values())][-10::]) * 100))

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
    print("execution time for each func",
          {k: [v[0], v[1] * total_exec_time]
           for k, v in sorted_func.items()})
    print("execution time percentage for each func", sorted_func)

    return top10_nodes


def block_analysis(step_log):
    sm_execution = {k: [] for k in step_log["SMs"].keys()}
    block_exec_time = {
        "blocks": {k: {}
                   for k in step_log["SMs"].keys()},  # horizontal
        "nodes": {k: {}
                  for k in step_log["SMs"].keys()},  # vertical
    }

    for sm in step_log["SMs"]:
        for (offset, nodeID, blockID), [start,
                                        end] in step_log["SMs"][sm].items():
            # confirm clock does proceed within an SM
            assert end > start
            assert start > step_log["sm_base_avg"][sm]

            start, end = [
                i - step_log["sm_base_avg"][sm] for i in [start, end]
            ]
            sm_execution[sm].append((start, end))

            # funcID = step_log["mapping"][nodeID][0]
            # hard coding for narrowphase
            # if funcID == 28:
            #     if offset not in FUNC_OFFSET_RANKINGS:
            #         FUNC_OFFSET_RANKINGS[offset] = end - start
            #     else:
            #         FUNC_OFFSET_RANKINGS[offset] += end - start

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

    sm_idle = []
    for s in sm_execution:
        intervals = []
        for start, end in sm_execution[s]:
            if len(intervals) == 0:
                intervals.append((start, end))
                continue
            p = 0
            while p < len(intervals):
                if start > intervals[p][1]:
                    p += 1
                    continue
                elif end < intervals[p][0]:
                    intervals.insert(p, (start, end))
                    break
                else:
                    intervals[p] = (min(start, intervals[p][0]),
                                    max(end, intervals[p][1]))
                    break
            else:
                intervals.insert(p, (start, end))
        gap = 0
        last_end = 0
        for start, end in intervals:
            gap += start - last_end
            last_end = end
        sm_idle.append(gap)

    sm_active = sorted([1 - i / step_log["final_timestamp"] for i in sm_idle])
    # print(sm_active)
    print(
        "For each SM on average, {:.3f}% of the time there is at least one block is running on"
        .format(sum(sm_active) / len(sm_active) * 100))

    return block_exec_time


COLORS = [
    "blue", "orange", "red", "green", "purple", "cyan", "pink", "yellow",
    "black", "black"
]


def plot_events(step_log, nodes, blocks, file_name):
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
    print("color mapping for functions:", colors)

    img = Image.new("RGB", (x_limit, y_limit), "white")
    draw = ImageDraw.Draw(img)

    def cast_coor(timestamp, limit=x_limit):
        assert (timestamp <= step_log["final_timestamp"])
        return int(timestamp / step_log["final_timestamp"] * limit)

    narrow_gap = {}

    for s, b in blocks.items():
        y = s * num_pixel_per_sm
        for bb, events in b.items():
            draw.line((cast_coor(step_log["final_cycles"][bb]), y, x_limit, y),
                      fill="grey",
                      width=1)
            for e in events:

                # to measure the gap between block events of node 150, function 28, narrowphase
                if e[2] == 150:
                    if bb not in narrow_gap:
                        narrow_gap[bb] = [(e[0], e[1])]
                    else:
                        narrow_gap[bb].append((e[0], e[1]))

                bar_color = colors[step_log["mapping"][
                    e[2]]] if step_log["mapping"][e[2]] in colors else "black"
                draw.line((cast_coor(e[0]), y, cast_coor(e[1]), y),
                          fill=bar_color,
                          width=2)
                # lighten the first pixel to indicate starting
                draw.line(
                    (cast_coor(e[0]), y, cast_coor(e[0]), y + 1),
                    fill=tuple(
                        (i + 255) // 2 for i in ImageColor.getrgb(bar_color)))
            y += 2

    idle_rate = []
    for _, v in narrow_gap.items():
        idle_time = 0
        last_end = v[0][1]
        for s, e in v[1:]:
            assert s > last_end
            idle_time += s - last_end
            last_end = e
        idle_rate.append(idle_time / (v[-1][1] - v[0][0]))
    print(
        sum(idle_rate) / len(idle_rate),
        "of the running time of node 150 (func id 28, narrowphase), blocks are not doing real tasks"
    )

    # mark the start and the end of major nodes
    for _, v in nodes.items():
        draw.line((cast_coor(v[1]), 0, cast_coor(v[1]), y_limit),
                  fill="red",
                  width=1)
        draw.line((cast_coor(v[2]), 0, cast_coor(v[2]), y_limit),
                  fill="green",
                  width=1)

    img.save(file_name)


def step_analysis(step=5, file_name="megakernel_events.png"):
    step_log = LOG_STEPS[step]

    variance = [
        max(v.values()) - min(v.values())
        for v in step_log["base_cycles"].values()
    ]
    print("step #", step, " base cycle variance on each sm:", variance)

    # step_log["sm_base_avg"] = {
    #     k: sum(v.values()) / len(v.values())
    #     for k, v in step_log["base_cycles"].items()
    # }
    # for global timing, no per sm calibration needed anymore
    min_time = min(min(i.values()) for i in step_log["base_cycles"].values())
    step_log["start_timestamp"] = min_time
    step_log["sm_base_avg"] = {
        k: min_time
        for k in step_log["base_cycles"].keys()
    }

    # manually add the nodeStart event for node 0
    step_log["events"][0][1] = (0, 0, step_log["sm_base_avg"][0])

    node_exec_time = serialized_analysis(step_log)

    block_exec_time = block_analysis(step_log)
    plot_events(step_log, node_exec_time, block_exec_time["blocks"], file_name)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        print(
            "python parse_device_tracing.py [log_name] [# stps, default 1] [start from, default 10]"
        )
        exit()

    with open(sys.argv[1], 'rb') as f:
        events = bytearray(f.read())
        assert len(events) % 32 == 0
        print("# logged events", len(events) // 32)
        for i in range(0, len(events), 32):
            get_device_log(events[i:i + 32])

    # default value
    steps = 1
    start_from = 10
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
    if len(sys.argv) == 4:
        start_from = int(sys.argv[3])

    for s in LOG_STEPS:
        LOG_STEPS[s]["final_timestamp"] -= LOG_STEPS[s]["start_timestamp"]
        for b in LOG_STEPS[s]["final_cycles"]:
            LOG_STEPS[s]["final_cycles"][b] -= LOG_STEPS[s]["start_timestamp"]

    dir_path = sys.argv[1] + "_megakernel_events"
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.mkdir(dir_path)
    # todo: limit
    for s in range(start_from, start_from + steps):
        step_analysis(s, dir_path + "/step{}.png".format(s))

    # visualize execution time distribution for narrowphase
    # FUNC_OFFSET_RANKINGS = {
    #     k: v / sum(v for v in FUNC_OFFSET_RANKINGS.values())
    #     for k, v in FUNC_OFFSET_RANKINGS.items()
    # }
    # x = sorted(list(FUNC_OFFSET_RANKINGS.keys()))
    # y = [FUNC_OFFSET_RANKINGS[xx] for xx in x]
    # plt.plot(x, y)
    # plt.savefig("distribution.png")

    # todo: aggregated analysis
