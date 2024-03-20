import os
import argparse
import cv2
import numpy as np
import rospy
from mixed_reality.projections import Projections
from robot.motion_planner import Motion_Planner

from llm_utils import LLM_UTILS
from sensor_msgs.msg import Image
from context.context4task import *
from context.core import *

from utils import *
from config import *
from cv_bridge import CvBridge
import hashlib
from pynput import keyboard
from pynput.keyboard import Key
import PySimpleGUI as sg
import json


class EXPERIMENT(Motion_Planner):
    
    def __init__(self, proj_x = 1.6, proj_y = 0.8, model="4.0-turbo", debug_mode = 1, seed = None, data_dir="/ssd_2/iros2024_data/user_exp_data", nli_duration = 240, debug = False,
                 move_only=False, same_user = False):
        super().__init__()
        # super().__init__()
        self.debug = debug
        self.move_only = move_only
        self.same_user = same_user
        if self.move_only: print(f"\nATTENTION: Robot will only move to waypoints but will not grab objects\n")
        
        self.data_dir = data_dir
        path = Path(__file__).parent.absolute()
        self.data_dir = path / self.data_dir
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.users = os.listdir(self.data_dir)
        self.users.sort()
        self.init_exp_paras = True
        
        if self.same_user:
            self.user_name = self.get_username()
        else:
            self.user_name = None
            self.init_paras()
            if self.user_name == None or self.user_name == '':
                self.user_name = "default"
            user_file= open(".username.txt","w")
            user_file.write(self.user_name)
            print(f"User : {self.user_name}")
    
        if self.same_user: print(f"\nATTENTION: loading saved user name: {self.user_name}\n")
        
        self.user_dir = self.data_dir  / self.user_name
        if not os.path.isdir(self.user_dir):
            os.makedirs(self.user_dir)
        self.svg_images_dir = PATHS.get("svg_images_dir")

        self.start_time = 0
        self.end_time = 0
        self.start_ts = None
        self.end_ts = None
        seed_seq =[int(ord(_)) for _ in self.user_name]
        np.random.seed(seed_seq)
          
        if self.debug:
            if debug_mode == 1:
                self.modes = ["nli"] * 6
                
            elif debug_mode == 2: 
                self.modes = ["vss"] * 6
            
            elif debug_mode == 3: 
                self.modes = ["screen"] * 6
        else:
            self.modes = ['nli', 'vss', 'vss', 'screen', 'screen', 'nli']
            
        self.structures =  ['R', 'S', 'U', 'O', 'K', 'Z']
        np.random.shuffle(self.modes); np.random.shuffle(self.structures)
        print(f"Experiment modes order: {self.modes}")
        print(f"Experiment structures order: {self.structures}")
        
        self.nli_counter = 1
        self.nli_duration = nli_duration  #secs
        self.proj_x = proj_x; self.proj_y = proj_y
        self.project = Projections(proj_x_len=self.proj_x, proj_y_len=self.proj_y)
        self.llms = LLM_UTILS(model=model)
        
        self.key = None
        self.keyboard_thread = Thread(target=self.key_listen, args=())
        self.keyboard_thread.start()
        
        self.img_dir = self.user_dir / f"images"
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.start_saving_imsgs = False
        self.cap_th = Thread(target=self.cv2_image_callback,args=())
        self.cap_th.start()
        return
    
    def get_username(self):
        if not os.path.isfile(".username.txt"):
            raise ValueError("User name is not set or write user name in \".username.txt\" file")
        
        try:
            with open(".username.txt","r") as fh:
                data = fh.read().splitlines()[0]
        except Exception as e:
            return None
        return data
    
    
    def cv2_image_callback(self):
        while not rospy.is_shutdown():
            if not self.start_saving_imsgs:
                continue
            if not os.path.isdir(self.img_dir):
                os.makedirs(self.img_dir)
            r, img = self.cap.read()
            # cv2.imshow("image",img)
            # cv2.waitKey(1)
            img_file = str(self.img_dir / f"{get_timestamp()}.jpg")
            cv2.imwrite(img_file, img)
            rospy.sleep(1/60)
        return 
    
    def key_listen(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()
        return
    
    def on_press(self, key):
        # print(self.previous_called)
        try:
            # End Experiment
            if key == Key.space:
                print("pressed \"space\"")
                self.end_time = time.time_ns()
                self.end_ts = get_timestamp()
                self.key = "space"    
                                        
            # Start Experiment
            if key.char == 's':
                print(f"pressed s")
                
                self.key = "s"
                        
        except AttributeError:
            pass
        return
    
    def init_paras(self):
        # All the stuff inside your window.
        layout = [ [sg.Text('Enter your name below')],
                   [sg.InputText(key='Input')],
                   [sg.Button('Ok', key="ok")] ]

        # Create the Window
        window = sg.Window('Human Robot Teaming Experiment', layout, size=(900, 500), font = ("Areial", 35), finalize=True,
                           location=(1080,0))
        window['Input'].bind("<Return>","_Enter")
        values = None
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            self.user_name = values.get("Input")
            if event == sg.WIN_CLOSED or event == 'Cancel' or event == "ok": # if user closes window or clicks cancel
                break
            if event == "Input" + "_Enter":
                print(f"You pressed \'Enter\'")
                break
            
        if values.get("Input") is None:
            self.terminate_th = True
            raise ValueError("Please enter the user name")

        window.close()
        self.init_exp_paras = False
        return 
    
    def synthesize_signals(self, properties,temperature = 0.0, max_token = 16000, obj_=True, inst_=True, nli_=True):
        # print(f"properties: {properties}")
        # task manager prompt and resp
        task_context = firstExperimentTaskDescription(structure          = properties.get("structure"),
                                                      object_description = properties.get("object_description"),
                                                      object_color       = properties.get("object_color"),
                                                      orientation        = properties.get("orientation"),
                                                      instruction        = properties.get("instruction"),
                                                      goal_position      = properties.get("goal_position"))
        tm_prompt = generate_task_master_prompt(task_context)
        # print(tm_prompt)
        tm_msg = self.llms.generate_openai_message(prompt=tm_prompt, printout=False)
        tm_rsp = self.llms.call_model_response(messages=tm_msg, temperature=temperature, max_tokens=max_token)
        
        if not isinstance(tm_rsp,str):
            tm_rsp = tm_rsp.choices[0].message.content
        
        if obj_:
        # obect visual signal prompt and resp
            obj_vss_prompt = generate_obj_vss_prompt(response=tm_rsp)
            # print(f"object vss prompt: \n{obj_vss_prompt}")
            obj_vss_msg = self.llms.generate_openai_message(prompt=obj_vss_prompt, printout=False)
            obj_vss_rsp = self.llms.call_model_response(messages=obj_vss_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(obj_vss_rsp,str):
                obj_vss_rsp = obj_vss_rsp.choices[0].message.content
        else:
            obj_vss_rsp = None

        if inst_:
            # instruction  visual signal prompt and resp
            inst_vss_prompt = generate_inst_vss_prompt(response=tm_rsp)
            inst_vss_msg = self.llms.generate_openai_message(prompt=inst_vss_prompt, printout=False)
            inst_vss_rsp = self.llms.call_model_response(messages=inst_vss_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(inst_vss_rsp,str):
                inst_vss_rsp = inst_vss_rsp.choices[0].message.content
        else:
            inst_vss_rsp = None
            
        if nli_:
            # natural language resp
            nlis_prompt = generate_nlis_prompt(response=tm_rsp)
            nlis_msg = self.llms.generate_openai_message(prompt=nlis_prompt, printout=False)
            nlis_rsp = self.llms.call_model_response(messages=nlis_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(nlis_rsp,str):
                nlis_rsp = nlis_rsp.choices[0].message.content
        else:
            nlis_rsp = None
        
        # print(f"object vss: {obj_vss_rsp}")
        # print(f"task master response: \n{tm_rsp}")
        return obj_vss_rsp, inst_vss_rsp, nlis_rsp
    
    
    
    def raterize_projection_visual_signals(self, obj_vss_rsp, inst_vss_rsp, object_name = None):
        # get image from text svg code
        inst_img = svg_to_cv(svg_code=get_svg_files(inst_vss_rsp)[0], show=False)
        obj_img_init = svg_to_cv(svg_code=get_svg_files(obj_vss_rsp)[0], show=False)
        save_svg(f"{str(self.svg_images_dir)}/{object_name}.svg",get_svg_files(obj_vss_rsp)[0])
        save_svg(f"{str(self.svg_images_dir)}/{object_name}_inst.svg",get_svg_files(inst_vss_rsp)[0])
        cv2.imwrite(f"{str(self.svg_images_dir)}/{object_name}.jpg", obj_img_init)
        iih, iiw, _ = inst_img.shape

        # parser transformtion instructions from inst
        (start_x, start_y), (goal_x, goal_y), ori = get_start_goal_orientation(inst_vss_rsp)
        icon_size = ICON_RES
        cv_icon_ctr_x = icon_size[0] // 2
        cv_icon_ctr_y = icon_size[1] // 2
        M = cv2.getRotationMatrix2D((cv_icon_ctr_x, cv_icon_ctr_y), -ori, 1.0)
        obj_img_rot = cv2.warpAffine(obj_img_init, M, (icon_size[1], icon_size[0]))
        
        outer_image = np.ones((int(1000 * self.proj_y), int(1000 * self.proj_x), 3), dtype=np.uint8) * 0
        copy_outer_image_1 = np.copy(outer_image)
        copy_outer_image_2 = np.copy(outer_image)
        copy_outer_image_3 = np.copy(outer_image)
        oih, oiw, _ = outer_image.shape
        x_offset = int((outer_image.shape[1] - inst_img.shape[1]) / 2)
        y_offset = int((outer_image.shape[0] - inst_img.shape[0]) / 2)
        
        start_x = start_x + x_offset
        start_y = start_y + y_offset
        goal_x = goal_x + x_offset
        goal_y = goal_y + y_offset
        copy_outer_image_1[start_y - cv_icon_ctr_y: start_y + cv_icon_ctr_y,
                           start_x - cv_icon_ctr_x: start_x + cv_icon_ctr_x,
                           :] = obj_img_init
        
        copy_outer_image_2[int((oih - iih) / 2):-int((oih - iih) / 2),
                           int((oiw - iiw) / 2):-int((oiw - iiw) / 2),
                           :] = inst_img

        copy_outer_image_3[goal_y - cv_icon_ctr_y: goal_y + cv_icon_ctr_y,
                           goal_x - cv_icon_ctr_x: goal_x + cv_icon_ctr_x,
                           :] = obj_img_rot
        
        visual_signals = [outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3, x_offset, y_offset]
        
        return visual_signals
        
    def raterize_screen_visual_signals(self, obj_vss_rsp, inst_vss_rsp, objects, object_goal_rot, object_goal_poses, object_name = None):
        outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3, x_offset, y_offset =  self.raterize_projection_visual_signals(obj_vss_rsp,inst_vss_rsp, object_name=object_name)
        icon_size = ICON_RES
        cv_icon_ctr_x = icon_size[0] // 2
        cv_icon_ctr_y = icon_size[1] // 2
        _outer_image = np.copy(outer_image)
        objects = objects[:-1]
        for oid, obj in enumerate(objects):
            tmp_img = np.copy(outer_image)
            img = cv2.imread(f"{str(self.svg_images_dir)}/{obj}.jpg")
            if img.shape[0] != icon_size[0]:
                img = cv2.resize(img,icon_size)
            if object_goal_rot[oid] != 0:
                M = cv2.getRotationMatrix2D((cv_icon_ctr_x, cv_icon_ctr_y), -object_goal_rot[oid], 1.0)
                img = cv2.warpAffine(img, M, (icon_size[1], icon_size[0]))
            _x =  object_goal_poses[oid][0] + x_offset
            _y =  object_goal_poses[oid][1] + y_offset
            tmp_img[_y - cv_icon_ctr_y: _y + cv_icon_ctr_y,
                    _x - cv_icon_ctr_x: _x + cv_icon_ctr_x,
                    :] = img
            # cv2.imshow(obj,img)
            _outer_image = cv2.addWeighted(_outer_image,1,tmp_img,1,1.0)
        visual_signals = [outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3]
        return visual_signals, _outer_image 
    
    def display_singal(self,visual_signals,_outer_image = None):
        alpha1 = 0.0
        alpha2 = 0.0
        alpha3 = 0.0
        delta  = 0.08
        while alpha1 < 1.0 or alpha2 < 1.0 or alpha3 < 1.0:
            proj_image = animated_image(visual_signals,
                                        alpha1, alpha2, alpha3)        
            if _outer_image is not None:
                proj_image_ =  cv2.addWeighted(np.copy(_outer_image),1,proj_image,1,1.0)
                cv2.namedWindow("screen",cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("screen",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow("screen",1080,0)
                cv2.imshow("screen",proj_image_)     
                                
            else:
                proj_image_ = cv2.rotate(proj_image,cv2.ROTATE_180)   
                warped = self.project.adjust_and_warp(proj_image_, rotate_c_90=True)
                self.project.show_on_projector(image=warped)
                # cv2.imshow("debug",proj_image)
            key = cv2.waitKey(1)
            alpha1 += delta
            alpha2 += delta
            alpha3 += delta                   
        return
    
    def display_nli(self,nlis_rsp):
        text = nlis_rsp.replace("```","").split("\n")
        # print(text)
        filtered_text = ""
        layout = []
        # print(text)
        counter = 0
        for tid , t in enumerate(text):
            if len(t) == 0:continue
            t = t.replace("\n","")
            filtered_text = t+"\n\n"
            txt_size = len(filtered_text)
            if counter == 0:
                layout.append([sg.HSeparator()])
                
            layout.append([sg.Text(text=filtered_text,size=(txt_size,2), font=('Roboto bold',25), text_color="white",key="NLI")])
        
            layout.append([sg.HSeparator()])
            counter += 1
          
        
        window = sg.Window(f'NLI (USER: {self.user_name})', layout, grab_anywhere=True, size=(1920, 1080), font = ('Roboto bold',25), finalize=True, location=(1080,0), resizable=True)
        # window.maximize()
        print("starting loop")
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            # print("running nli loop")
            
            # event, values = window.read()
            # print(event, values)
            if self.key == "space": # if user closes window or clicks cancel
                print(f"closing window...")
                window.close()
                break
        # sg.popup_scrolled(filtered_text,
        #                   title="Natural Language Instructions",
        #                   font = ("Areial", 35), 
        #                   size = (100,100),
        #                   auto_close=True,
        #                   auto_close_duration=self.nli_duration,
        #                   relative_location=(0,0),location=(1080,0)
        #                 )
        return 
    
    def save_time(self, start_time, end_time, start_ts, end_ts, mode, id):        
        ts = get_timestamp()
        start_time_list=  [int(t) for t in start_ts.split("_")]
        end_time_list=  [int(t) for t in end_ts.split("_")]
        file_name = self.user_dir / f"{ts}_{mode}_{id}.json"
        data = {"start_time":start_time, 
                "end_time":end_time, 
                "start_time_list":start_time_list, 
                "end_time_list":end_time_list}
        with open(file_name,"w",  encoding='utf-8') as fh:
            json.dump(data,fh,indent=4)
        return True         

    def main(self, skip_seq = 7):
        if not self.debug: self.go_home_v2(type=1)
        
        for seq_id, exp_args in enumerate(zip(self.modes,self.structures),1):
            
            
            mode, structure = exp_args 
            if seq_id <= skip_seq:
                print(f"skipping structure: {structure}")
                continue
            print("press \"s\" to start experiment!...")            
            while self.key != "s":                
                continue
            
            
            print(f"\n[MODE]: {mode}")
            start_waypoints, end_waypoints, object_desc, object_color, skip_id, orientation, instructions, goal_poses = self.get_waypoints(shape=structure)
            objects = []      
            object_goal_poses = [] 
            object_goal_rot   = []  
            for i, waypoint in enumerate(zip(start_waypoints, end_waypoints)):
                start_waypoint_, end_waypoint = waypoint  
                start_waypoint = start_waypoint_[0]
                objects.append(start_waypoint_[1])
                object_goal_poses.append(goal_poses[i][0])
                object_goal_rot.append(goal_poses[i][1][1])
                goal_object_to_pick = start_waypoint_[1] 
                print(f"picking up: {goal_object_to_pick}")
                if skip_id == i:
                    self.img_dir = self.user_dir / f"images_{mode}_{seq_id}"
                    self.start_saving_imsgs = True
                    if not self.debug: self.go_home_v2(type=2)
                    
                    self.start_time = time.time_ns()
                    self.start_ts = get_timestamp()
                    
                    properties = {"structure"          : structure,
                                  "object_description" : object_desc,
                                  "object_color"       : object_color,
                                  "orientation"        : orientation[0],
                                  "instruction"        : instructions,
                                  "goal_position"      : goal_poses[i][0]}

                    # properties = {"structure"          : "S",
                    #               "object_description" : "Bunny",
                    #               "object_color"       : "Orange",
                    #               "orientation"        : "upright",
                    #               "instruction"        : "some zig zags from bottom to top",
                    #               "goal_position"      : [500,100]}
                    print(properties)
                    
                    obj_vss_rsp, inst_vss_rsp, nlis_rsp = self.synthesize_signals(properties=properties,temperature=0.0)
                    self.key = None
                    if mode == "vss":
                        # visual_signals = self.raterize_projection_visual_signals(obj_vss_rsp, inst_vss_rsp, object_name="Bunny")
                        visual_signals = self.raterize_projection_visual_signals(obj_vss_rsp, inst_vss_rsp, object_name=object_desc)
                        while True:
                            self.display_singal(visual_signals[:-2])
                            if self.key == "space":
                                cv2.destroyAllWindows()
                                break
                            rospy.sleep(1)
                            
                    elif mode == "screen":
                        visual_signals, _outer_image = self.raterize_screen_visual_signals(obj_vss_rsp, inst_vss_rsp, objects, object_goal_rot, object_goal_poses,
                                                                                           object_name=object_desc)
                        while True:
                            self.display_singal(visual_signals, _outer_image)
                            if self.key == "space":
                                cv2.destroyAllWindows()
                                break
                            rospy.sleep(1)
        
                    elif mode == "nli":
                        self.display_nli(nlis_rsp)
                        
                    if self.end_time < self.start_time:
                        self.end_time = time.time_ns()
                        self.end_ts = get_timestamp()
                    self.save_time(start_time=self.start_time,end_time=self.end_time,
                                   start_ts=self.start_ts, end_ts = self.end_ts, mode=mode, id = seq_id)
                    self.start_saving_imsgs = False
                    continue
                
                if self.debug:
                    continue
        
                for wid, w in enumerate(start_waypoint):
                    # print(f"wid: {wid} and start waypoint: {w}")
                    if wid == 3:
                        break
                    goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions,xyz=w[:3],rpy = w[3:6] )
                    self.movej_msg(goal_joints=goal_joints,t=w[6])
                    if bool(w[7]) and not self.move_only:
                        self.r2fg_msg.position = self.max_r2fg
                    else:
                        self.r2fg_msg.position = 0
                    self.ur5_joint_publisher.publish(self.ur5_control_msg)
                    self.r2fg_publisher.publish(self.r2fg_msg)
                    rospy.sleep(w[6])

                for wid, w in enumerate(end_waypoint):
                    # print(f"wid: {wid} and end waypoint: {w}")
                    if wid == 3:
                        break
                    goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions,xyz=w[:3],rpy = w[3:6] )
                    self.movej_msg(goal_joints=goal_joints,t=w[6])
                    if bool(w[7]) and not self.move_only:
                        self.r2fg_msg.position = self.max_r2fg
                    else:
                        self.r2fg_msg.position = 50

                    self.ur5_joint_publisher.publish(self.ur5_control_msg)
                    self.r2fg_publisher.publish(self.r2fg_msg)
                    rospy.sleep(w[6])
            if not self.debug: self.go_home_v2(type=1)
        print(f"Finished experiment!!!!!!!!!!")  
        return
    
    def end_thread(self):
        pass
    
if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_seq", type=int, default= 0, help="number of structures to skip")
    parser.add_argument("--debug", type=int, default= 0, help="number of structures to skip")
    parser.add_argument("--debug_mode", type=int, default= 1, help="1: nli, 2: vss, 3: screen")
    parser.add_argument("--move_only", type=int, default= 0, help="move robot but dont grab")
    parser.add_argument("--same_user", type=int, default= 0, help="load same user")
    args = parser.parse_args()
    exp = EXPERIMENT(debug=bool(args.debug), debug_mode=args.debug_mode, move_only=bool(args.move_only), same_user=bool(args.same_user))
    exp.main(skip_seq=args.skip_seq)
    exp.end_thread()