import re
import os
import sys
import cv2
from cairosvg import svg2png
from text2svg import *
from datetime import datetime
from functools import lru_cache
import hashlib
import pickle
import subprocess
import time
import numpy as np
import config

cache_dir = config.PATHS["cache_dir"]

def time_taken(func):
    def wrappper(*args, **kwargs):
        t1 = time.time_ns()
        output = func(*args, **kwargs)
        t2 = time.time_ns()
        print(f"time taken by function {func.__name__} = {(t2 - t1) * 1e-9} seconds")
        return output

    return wrappper


def hash_it(content):
    if content is None:
        return
    encoder = hashlib.sha256()
    encoder.update(bytes(content, 'utf-8'))
    hashed = encoder.hexdigest()
    return hashed


def get_cache_item(hashed):
    with open(os.path.join(cache_dir, hashed), "rb") as fh:
        data = pickle.load(fh)
    return data


def write_cache_item(hashed, content):
    with open(os.path.join(cache_dir, hashed), "wb") as fh:
        pickle.dump(content, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return


def naive_cache(func):
    avail_cache = os.listdir(cache_dir)

    def wrapper(*args, **kwargs):
        model = kwargs.get("model")
        temperature = kwargs.get("temperature")
        messages = kwargs.get("messages")

        if messages != None and model != None:
            # encoder = hashlib.sha256()
            content = f"model: {model}\ntemperature: {temperature}\nmessages: {messages}"
            # encoder.update(bytes(content, 'utf-8'))
            # hashed = encoder.hexdigest()
            hashed = hashlib.sha256(content.encode('utf-8')).hexdigest()
            # print(f"\n\n\n\n\n")
            # print(f"input for hash:{content}")
            # print(f"hashcode: {hashed}")
            # print(f"\n\n\n\n\n")
            
            if hashed in avail_cache:
                print(f"item available in cache")
                content, response = get_cache_item(hashed)
            else:
                print(f"new item added to the cache")
                response = func(*args, **kwargs)
                write_cache_item(hashed, [content, response])
            return response
        else:
            raise ValueError("Provide values for following arguments:\n\n\'message\'\n\'model\'\n\'temperature\'\n")

    return wrapper


def get_timestamp():
    return str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")


def get_svg_files(response, re_pattern=r'<svg.*?</svg>'):
    svg_matches = re.findall(re_pattern, response, re.DOTALL)
    return svg_matches


def get_start_goal_orientation(inst_vss_rsp) -> ((int, int), (int, int), int):
    """returns: start[x,y], goal[x,y], orientation"""
    start = re.findall(r'<start_position.*?</start_position>', inst_vss_rsp, re.DOTALL)[0]
    start_x = int(re.findall(r'\[.*?,', start, re.DOTALL)[0][1:-1])
    start_y = int(re.findall(r',.*?]', start, re.DOTALL)[0][1:-1])
    goal = re.findall(r'<goal_position.*?</goal_position>', inst_vss_rsp, re.DOTALL)[0]
    goal_x = int(re.findall(r'\[.*?,', goal, re.DOTALL)[0][1:-1])
    goal_y = int(re.findall(r',.*?]', goal, re.DOTALL)[0][1:-1])
    ori = int(re.findall(r'<orientation.*?</orientation>', inst_vss_rsp, re.DOTALL)[0][13:-14])
    return (start_x, start_y), (goal_x, goal_y), ori


def get_png_file(response, re_pattern=r'\'\'\'png\'\'\'.*?png\'\'\''):
    png_matches = re.findall(re_pattern, response, re.DOTALL)


def get_python_f(response, re_pattern=r'\`\`\`python.*?\`\`\`'):
    return re.findall(re_pattern, response, re.DOTALL)[0].replace("python", "").replace("```", "")


def svg_to_cv(svg_code, filename=".output.png", h=500, w=500, show=True):
    try:
        # print(svg_code)
        svg2png(bytestring=svg_code, write_to=filename)
        # print(filename)
        img = cv2.imread(filename)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.rotate(img, cv2.ROTATE_180)

        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img = cv2.flip(img,0)

        # img = cv2.resize(img, (h,w), interpolation=cv2.INTER_LINEAR_EXACT)
        if show:
            cv2.imshow(f"image", img)
            cv2.waitKey(-1)
        return img
    except Exception as e:
        print(f"Got exception in svg to cv function!\nException: {e}")


def get_nl_instructions(response, pattern=r"\[Natural Language Instructions\](.*?)----"):
    nl_instructions = re.findall(pattern, response, re.DOTALL)[0]
    # nl_instructions = nl_instructions.split("\n-")
    nl_prefix = f"Please follow instructions below:\n"
    nl_instructions = nl_prefix + nl_instructions
    # for id, nli in enumerate(nl_instructions):
    #     # nli = nli.replace("\n- ","")
    #     if id == 0:
    #         continue
    #     print(f"{id}. {nli}")
    filename = f"/tmp/instructions_{get_timestamp()}.txt"
    with open(filename, "w") as fh:
        fh.write(nl_instructions)
    command = ["xdg-open", filename]
    subprocess.call(command)
    print(nl_instructions)
    return nl_instructions


def get_goal_obj_prompt(response, pattern=r"\[Goal Objects\](.*?)----"):
    nl_instructions = re.findall(pattern, response, re.DOTALL)[0]
    nl_instructions = nl_instructions.replace("Prompt = ", "")
    return nl_instructions


def get_caution_prompt(response, pattern=r"\[Caution\](.*?)\[/Caution\]"):
    nl_instructions = re.findall(pattern, response, re.DOTALL)[0]
    nl_instructions = nl_instructions.replace("Prompt = ", "")
    return nl_instructions


def get_rel_info_prompt(response, pattern=r"\[Relational Information\](.*?)----"):
    nl_instructions = re.findall(pattern, response, re.DOTALL)[0]
    nl_instructions = nl_instructions.replace("Prompt = ", "")
    # nl_instructions = nl_instructions.split("\n-")
    # nl_prefix = f"Please follow instructions as below:\n"
    nl_prefix = f"\n"
    nl_instructions = nl_prefix + nl_instructions
    # for id, nli in enumerate(nl_instructions):
    #     # nli = nli.replace("\n- ","")
    #     if id == 0:
    #         continue
    #     print(f"{id}. {nli}")
    # filename = f"/tmp/instructions_{get_timestamp()}.txt"
    # with open(filename,"w") as fh:
    #     fh.write(nl_instructions)
    # command = ["xdg-open",filename]
    # subprocess.call(command)
    # print(nl_instructions)
    return nl_instructions


def all_none(go_ids):
    counter = 0
    for gid in go_ids:
        if gid == '':
            counter += 1

    if len(go_ids) == counter:
        return True
    else:
        return False

def save_svg(filename,code):
    with open(filename,"w") as fh:
        fh.write(code)
    return 

def get_env_and_goal_obj_pose_info(response):
    env_pattern = r"\[Environment\](.*?)----"
    env_info = re.findall(env_pattern, response, re.DOTALL)
    for id in env_info:
        id = id.split("\n")
        for element in id:
            if "Grid Size" in element:
                gsize = element.replace("Grid Size = [", "").replace("]", "").split(",")
                gsize = [int(gsize[0]), int(gsize[1])]
                print(f"Grid Size: {gsize}")

            if "Background Color" in element:
                color = element.replace("Background Color = [", "").replace("]", "").split(",")
                color = [int(c) for c in color]
                print(f"Background color: {color}")
    goal_obj_pose_pattern = r"\[Goal Object Positions\](.*?)----"
    goal_obj_info = re.findall(goal_obj_pose_pattern, response, re.DOTALL)[0]
    go_ids = goal_obj_info.split(";")
    # print(go_ids)
    goal_obj_poses = []
    for id, go_id in enumerate(go_ids):
        obj_pose = []
        go_id = go_id.split("\n")
        if all_none(go_id):
            continue
        for element in go_id:
            if "Start Position" in element:
                pose = element.replace("Start Position = [", "").replace("]", "").split(",")
                pose = [int(p) for p in pose]
                obj_pose.append(pose)

            if "End Position" in element:
                pose = element.replace("End Position = [", "").replace("]", "").split(",")
                pose = [int(p) for p in pose]
                obj_pose.append(pose)

        goal_obj_poses.append(obj_pose)
        # print(f"goal id: {id} and pose: {go_id}")
    #     go_id = go_id.split("\n")
    #     for element in go_id:
    #         element = element.split(",")
    #         obj_pose = []
    #         for poses in element:
    #             print(poses)
    #             # if "Start Position" in poses:
    #             #     pose = poses.replace("Start Position = [","").replace("]","").split(",")
    #             #     pose = [int(p) for p in pose]
    #             #     print(pose)
    #             #     obj_pose.append(pose)

    #             # if "End Position" in poses:
    #             #     pose = poses.replace("End Position = [","").replace("]","").split(",")
    #             #     pose = [int(p) for p in pose]
    #             #     obj_pose.append(pose)
    #             # goal_obj_poses.append(obj_pose)
    # print(goal_obj_poses)
    return gsize, color, goal_obj_poses



def animated_image(visual_signals,
            alpha1 = 0.0, alpha2 = 0.0, alpha3 = 0.0, delta = 0.08):
    
    outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3 = visual_signals                  
    if alpha1 < 1.0:
        complete_image = cv2.addWeighted(outer_image, 1, copy_outer_image_1, alpha1, 1.0)
        first_step_image  = np.copy(complete_image).astype(np.uint8)
        alpha1 += delta
        
    if alpha2 < 1.0 and alpha1 >= 1.0:
        second_image = cv2.addWeighted(outer_image, 1, copy_outer_image_2, alpha2, 1.0)
        complete_image = cv2.addWeighted(first_step_image, 1, second_image, 1.0, 1.0)
        second_step_image  = np.copy(complete_image).astype(np.uint8)
        alpha2 += delta
    
    if alpha3 < 1.0  and alpha1 >= 1.0 and alpha2 >= 1.0:
        third_image = cv2.addWeighted(outer_image, 1, copy_outer_image_3, alpha3, 1.0)
        complete_image = cv2.addWeighted(second_step_image, 1, third_image, 1.0, 1.0)
        alpha3 += delta   
    return complete_image