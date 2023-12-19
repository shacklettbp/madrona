#!/usr/bin/env python3

import sys
import os

def read_config(cfg_path):
    instance_funcs = []
    device_funcs = []

    cur_funcs = None
    cur_cond_var = None
    cond_space = None
    with open(cfg_path, 'r') as cfg:
        err = False
        for line in cfg:
            noprefix = line.lstrip()
            leading_ws = len(line) - len(noprefix)

            if cur_cond_var != None:
                if cond_space == None:
                    if leading_ws == 0:
                        err = True
                        break
                    else:
                        cond_space = leading_ws
                elif leading_ws < cond_space:
                    cur_cond_var = None
                    cond_space = None
                elif leading_ws > cond_space:
                    err = True
                    break
                
            line = noprefix.rstrip()
            if len(line) == 0:
                continue
            elif line[-1] == ':':
                if line[:-1] == 'instance':
                    cur_funcs = instance_funcs
                elif line[:-1] == 'device':
                    cur_funcs = device_funcs
                elif line[0] == '-' and cur_cond_var == None:
                    cur_cond_var = line[1:-1].strip()
                else:
                    err = True
                    break
            elif line[0] == '-':
                func = line[1:].strip()
                if len(func) < 3 or func[:2] != 'vk':
                    err = True
                    break
                else:
                    if cur_cond_var != None:
                        cur_funcs.append((func, cur_cond_var))
                    else:
                        cur_funcs.append(func)
            else:
                err = True
                break

        if err:
            print(f"bad line: {line}")
            sys.exit(1)

    return instance_funcs, device_funcs

def gen_single_src(funcs, lookup_func):
    pad = ' '*4
    inst_hpp = ''
    inst_cpp = ''

    if len(funcs) > 0:
        inst_cpp = pad + ':\n'

    for i, func in enumerate(funcs):
        cond = ""
        if isinstance(func, tuple):
            func, cond_var = func
            cond = f"!{cond_var} ? nullptr :"

        member_name = func[2].lower() + func[3:]
        pointer_type = "PFN_" + func
        inst_hpp += f"{pad}{pointer_type} {member_name};\n"
        trail = ',' if i < len(funcs) - 1 else ''
        # Vulkan 1.2.193 vkGetInstanceProcAddr cannot be called with a valid instance as argument.
        # Just reuse the existing vkGetInstanceProcAddr pointer in this case
        if func == 'vkGetInstanceProcAddr':
            inst_cpp += f"{pad}  {member_name}({cond} {lookup_func}){trail}\n"
        else:
            inst_cpp += f"{pad}  {member_name}({cond} reinterpret_cast<{pointer_type}>(checkPtr({lookup_func}(ctx, \"{func}\"), \"{func}\"))){trail}\n"

    return inst_hpp, inst_cpp

def gen_src(instance_funcs, device_funcs):
    inst_hpp, inst_cpp = gen_single_src(instance_funcs, 'get_inst_addr')
    dev_hpp, dev_cpp = gen_single_src(device_funcs, 'get_dev_addr')

    return inst_hpp, inst_cpp, dev_hpp, dev_cpp


    
def generate_header(config_path, out_dir):
    inst_funcs, dev_funcs = read_config(config_path)
    inst_hpp, inst_cpp, dev_hpp, dev_cpp = gen_src(inst_funcs, dev_funcs)

    def write_src(src, name):
        with open(os.path.join(out_dir, name), 'w') as file:
            file.write(src)

    write_src(inst_hpp, 'dispatch_instance_impl.hpp')
    write_src(inst_cpp, 'dispatch_instance_impl.cpp')
    write_src(dev_hpp, 'dispatch_device_impl.hpp')
    write_src(dev_cpp, 'dispatch_device_impl.cpp')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"{sys.argv[0]}: CONFIG_PATH OUT_DIR", file=sys.stderr)
        sys.exit(1)

    generate_header(sys.argv[1], sys.argv[2])
