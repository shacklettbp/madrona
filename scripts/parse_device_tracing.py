import os
import sys
from PIL import Image, ImageDraw, ImageColor

LOG_STEPS = {}
NUM_HIGHLIGHT_NODES = 10


def parse_device_logs(events):
    global LOG_STEPS
    STEP = -1

    def new_step():
        nonlocal STEP
        STEP += 1
        LOG_STEPS[STEP] = {
            "events": {},
            "SMs": {},
            "mapping": {},
            "final_cycles": {},
            "start_timestamp": 0xFFFFFFFFFFFFFFFF,
            "final_timestamp": 0
        }

    for i in range(0, len(events), 32):
        array = events[i:i + 32]

        event = int.from_bytes(array[:4], byteorder='little')
        funcID = int.from_bytes(array[4:8], byteorder='little')
        numInvocations = int.from_bytes(array[8:12], byteorder='little')
        nodeID = int.from_bytes(array[12:16], byteorder='little')
        blockID = int.from_bytes(array[16:20], byteorder='little')
        smID = int.from_bytes(array[20:24], byteorder='little')
        cycleCount = int.from_bytes(array[24:32], byteorder='little')

        if event == 0:
            # beginning of a megakernel
            new_step()
            LOG_STEPS[STEP]["start_timestamp"] = cycleCount

        elif event in [1, 2]:
            if nodeID not in LOG_STEPS[STEP]["mapping"]:
                LOG_STEPS[STEP]["mapping"][nodeID] = (funcID, numInvocations)
            else:
                assert LOG_STEPS[STEP]["mapping"][nodeID] == (funcID,
                                                              numInvocations)

            if nodeID not in LOG_STEPS[STEP]["events"]:
                LOG_STEPS[STEP]["events"][nodeID] = {
                    event: (smID, blockID, cycleCount)
                }
            else:
                assert event not in LOG_STEPS[STEP]["events"][nodeID]
                LOG_STEPS[STEP]["events"][nodeID][event] = (smID, blockID,
                                                            cycleCount)

        elif event in [3, 4]:
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

        elif event == 5:
            assert blockID not in LOG_STEPS[STEP]["final_cycles"]
            LOG_STEPS[STEP]["final_cycles"][blockID] = cycleCount
            LOG_STEPS[STEP]["final_timestamp"] = max(
                LOG_STEPS[STEP]["final_timestamp"], cycleCount)
        else:
            assert (False & "event {} not supported".format(event))

    # drop the last step which might be corrupted
    # del LOG_STEPS[STEP]
    print(
        "At the end, complete traces for {} steps are generated".format(STEP))


def serialized_analysis(step_log):

    def calibrate(timestamp, reference):
        return timestamp[2] - reference

    node_exec_duration = {}
    for i in range(max(step_log["events"])):
        # skipped nodes
        if i not in step_log["events"]:
            continue
        node_exec_duration[i] = (
            *step_log["mapping"][i],
            calibrate(step_log["events"][i][2], step_log["start_timestamp"]) -
            calibrate(step_log["events"][i][1], step_log["start_timestamp"]))

    sorted_duration = dict(
        sorted(node_exec_duration.items(), key=lambda item: item[1][2]))

    total_exec_time = sum(v[2] for v in sorted_duration.values())
    normailized = {
        k: (v[0], v[1], round(v[2] / total_exec_time, 3))
        for k, v in sorted_duration.items()
    }
    print("Total execution time of the mega kernel: {:.4f}ms".format(
        total_exec_time / 1000000))
    # print("execution time percentage for each node", normailized)

    top_nodes = {
        i: (normailized[i][-1],
            calibrate(step_log["events"][i][1], step_log["start_timestamp"]),
            calibrate(step_log["events"][i][2], step_log["start_timestamp"]))
        for i in list(normailized.keys())[:-(NUM_HIGHLIGHT_NODES + 1):-1]
    }

    print("Top {} nodes amounts {:.3f}% of execution time".format(
        NUM_HIGHLIGHT_NODES,
        sum([i[-1] for i in list(normailized.values())
             ][-(NUM_HIGHLIGHT_NODES + 1)::]) * 100))

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
    # print("execution time for each func",
    #       {k: [v[0], v[1] * total_exec_time]
    #        for k, v in sorted_func.items()})
    # print("execution time percentage for each func", sorted_func)

    return top_nodes


def block_analysis(step_log):
    sm_execution = {k: [] for k in step_log["SMs"].keys()}
    block_exec_time = {
        "blocks": {k: {}
                   for k in step_log["SMs"].keys()},  # horizontal
        "nodes": {k: {}
                  for k in step_log["SMs"].keys()},  # vertical
    }

    for sm in step_log["SMs"]:
        for (offset, nodeID,
             blockID), time_stamps in step_log["SMs"][sm].items():
            start = time_stamps[0]
            end = max(time_stamps[1:])
            # confirm clock does proceed within an SM
            assert end >= start
            assert start > step_log["start_timestamp"]

            start, end = [
                i - step_log["start_timestamp"] for i in [start, end]
            ]
            sm_execution[sm].append((start, end))

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
    print(
        "For each SM on average, {:.3f}% of the time there is at least one block is running on"
        .format(sum(sm_active) / len(sm_active) * 100))

    return block_exec_time


COLORS = [
    "blue", "orange", "red", "green", "purple", "cyan", "pink", "magenta",
    "olive", "navy", "teal", "maroon", "yellow", "black"
]


def plot_events(step_log, nodes, blocks, file_name):
    num_sm = len(blocks)
    num_block_per_sm = 1
    num_pixel_per_block = 2
    sm_interval_pixel = 2
    num_pixel_per_sm = num_block_per_sm * num_pixel_per_block + sm_interval_pixel
    y_blank = 100
    y_limit = num_sm * num_pixel_per_sm + y_blank
    x_limit = y_limit * 3

    colors = {}
    for n in nodes:
        func = step_log["mapping"][n]
        if func not in colors and len(colors) < len(COLORS):
            colors[func] = COLORS[len(colors)]
    print("Color mapping for functions:", colors)

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
                bar_color = colors[step_log["mapping"][
                    e[2]]] if step_log["mapping"][e[2]] in colors else "black"
                draw.line((cast_coor(e[0]), y, cast_coor(e[1]), y),
                          fill=bar_color,
                          width=num_pixel_per_block)
                # lighten the first pixel to indicate starting
                draw.line(
                    (cast_coor(e[0]), y, cast_coor(
                        e[0]), y + num_pixel_per_block - 1),
                    fill=tuple(
                        (i + 255) // 2 for i in ImageColor.getrgb(bar_color)))
            y += sm_interval_pixel

    # mark the start and the end of major nodes
    y_shift = 0.9
    for n, v in nodes.items():
        left, right = cast_coor(v[1]), cast_coor(v[2])
        draw.line((left, 0, left, y_limit), fill="red", width=1)
        draw.line((right, 0, right, y_limit), fill="green", width=1)
        draw.text(
            (left, y_limit - y_blank * y_shift),
            " f: {}\n t: {:.3f}ms\n {:.1f}%".format(step_log["mapping"][n],
                                                    (v[2] - v[1]) / 1000000,
                                                    v[0] * 100),
            fill=(0, 0, 0))
        y_shift = 1.3 - y_shift

    img.save(file_name)


def step_analysis(step, file_name):
    step_log = LOG_STEPS[step]

    # manually add the nodeStart event for node 0
    step_log["events"][0][1] = (0, 0, step_log["start_timestamp"])

    node_exec_time = serialized_analysis(step_log)

    block_exec_time = block_analysis(step_log)
    plot_events(step_log, node_exec_time, block_exec_time["blocks"], file_name)


if __name__ == "__main__":
    if len(sys.argv) > 5:
        print(
            "python parse_device_tracing.py [log_name] [# stps, default 1] [start from, default 10] [# highlight nodes, default 10]"
        )
        exit()

    with open(sys.argv[1], 'rb') as f:
        events = bytearray(f.read())
        assert len(events) % 32 == 0
        print("{} events were logged in total".format(len(events) // 32))
        parse_device_logs(events)

    # default value
    steps = 5
    start_from = 10
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
    if len(sys.argv) >= 4:
        start_from = int(sys.argv[3])
    if len(sys.argv) >= 5:
        NUM_HIGHLIGHT_NODES = int(sys.argv[4])

    for s in LOG_STEPS:
        LOG_STEPS[s]["final_timestamp"] -= LOG_STEPS[s]["start_timestamp"]
        for b in LOG_STEPS[s]["final_cycles"]:
            LOG_STEPS[s]["final_cycles"][b] -= LOG_STEPS[s]["start_timestamp"]

    dir_path = sys.argv[1] + "_megakernel_events"
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.mkdir(dir_path)
    # todo: limit
    assert start_from < len(LOG_STEPS)
    for s in range(start_from, min(start_from + steps, len(LOG_STEPS))):
        step_analysis(s, dir_path + "/step{}.png".format(s))

    # todo: aggregated analysis
