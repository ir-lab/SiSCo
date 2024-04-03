#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from irl_robots.msg import ur5Control, gSimpleControl, ur5Joints
from transforms3d.euler import quat2euler, mat2euler

import os
import time
import json
from threading import Thread
import numpy as np

from robot.robot_kd_solver import Robot_KD_Solver
from robot.tasks import Tasks

import cv2
import random

HOME_JOINTS_RAD = np.array([2.0167279506794475, -1.8465583486100006, 1.743409389817136, -1.4702653618800232, -1.5704472609444977, 0.4436626958569586],dtype=float)
HOME_JOINTS_RAD_2 = np.array([2.0167129039764404, -2.4554107824908655, 2.3721654415130615, -1.4703057448016565, -1.5705207029925745, 0.4437211751937866],dtype=float)


class Motion_Planner(Robot_KD_Solver,Tasks):
    """
    Inherits from Robot_KD_Solver and Tasks to provide motion planning functionalities.

    This class handles the motion planning of a robot by utilizing the kinematic solvers
    from the Robot_KD_Solver and task-based functionalities from the Tasks class.

    It is designed to interact with ROS (Robot Operating System) for controlling a robot
    by publishing and subscribing to relevant topics."""
    
    def __init__(self) -> None:
        super(Motion_Planner,self).__init__()      
        rospy.init_node("motion_planner")
        self.shapes_ids = np.arange(6)
        random.shuffle(self.shapes_ids )
        self.strcture_id = 0
        rospy.set_param("FINISHED",False)
    
        self.grid = np.zeros((500,1000,3),dtype=np.uint8)
        self.x = 500
        self.y = 250
        
        self.ur5Joints = ur5Joints()
        self.ur5_control_msg = ur5Control()
        self.r2fg_msg        = gSimpleControl()
        self.r2fg_msg.force = 255
        self.r2fg_msg.speed = 255
        self.r2fg_max = 255
        self.max_r2fg = 110
        
        self.ur5_max_radius = 0.84 # meters
        self.z_min = -0.05
        self.z_max = 0.60
        self.y_min = 0.10
        self.y_max = 1.0
        self.d_xyz_max = 0.6
        self.timeout_maxiter = 10
        self.timeout_flag = False
        self.enable_flag = False
        self.sw_data = Float32MultiArray()
        
        self.ur5_control_msg.command      = rospy.get_param("irl_robot_command",  default="movej") 
        self.ur5_control_msg.acceleration = rospy.get_param("irl_robot_accl",     default=np.pi/2)
        self.ur5_control_msg.velocity     = rospy.get_param("irl_robot_vel",      default=np.pi/2) 
        self.ur5_control_msg.time         = rospy.get_param("irl_robot_com_time", default=5)
        self.ur5_control_msg.jointcontrol = True

        rospy.Subscriber("/ur5/joints",ur5Joints,self.ur5_joints_callback)
        
        self.ur5_joint_publisher = rospy.Publisher("/ur5/control",ur5Control,queue_size=1)
        self.r2fg_publisher = rospy.Publisher("/r2fg/simplecontrol",gSimpleControl,queue_size=1)
        self.control_freq = 50.0
        self.trigger_counter  = 0
        
        self.go_home_thread_flag = True
        self.go_home_thread = Thread(target=self.go_home,args=())
        return 
        
    def ur5_joints_callback(self,msg:ur5Joints) -> None:
        self.ur5Joints = msg

    
    def movej_msg(self, goal_joints:list, vel:float = 0.0, accel:float = 0.0,  t:float = 5.0, blend = 0.0) -> None:
        if len(goal_joints) != 6:
            raise ValueError("Please provide all joint angles...")   
        self.ur5_control_msg.command = "movej"
        self.ur5_control_msg.time    = t
        self.ur5_control_msg.acceleration = vel
        self.ur5_control_msg.velocity = accel
        self.ur5_control_msg.blend = blend
        self.ur5_control_msg.jointcontrol = True
        self.ur5_control_msg.values = goal_joints
        
    def movel_msg(self, goal_joints:list, vel:float = 0.0, accel:float = 0.0,  t:float = 5.0, blend = 0.0) -> None:
        if len(goal_joints) != 6:
            raise ValueError("Please provide all joint angles...")   
        self.ur5_control_msg.command = "movel"
        self.ur5_control_msg.time    = t
        self.ur5_control_msg.acceleration = vel
        self.ur5_control_msg.velocity = accel
        self.ur5_control_msg.blend = blend
        self.ur5_control_msg.jointcontrol = True
        self.ur5_control_msg.values = goal_joints
        
    
    def speedj_msg(self, goal_joints:list, t:float = 0.0, accl:float= 0.0) -> None:
        if len(goal_joints) != 6:
            raise ValueError("Please provide all joint angles...")
        self.ur5_control_msg.command = "speedj"
        self.ur5_control_msg.time = t
        self.ur5_control_msg.velocity = 0.0
        self.ur5_control_msg.acceleration = accl
        self.ur5_control_msg.jointcontrol = True
        self.ur5_control_msg.values = goal_joints

    def speedl_msg(self, d_xyz:list, d_rpy:list, accel:float = np.pi/4, t:float = 1.0) -> None:
        if len(d_xyz) != 3 and len(d_rpy) != 3:
            raise ValueError("Please check xyz and rpy values...")
        self.ur5_control_msg.command = "speedl"
        self.ur5_control_msg.time = t
        self.ur5_control_msg.velocity = 0.0
        self.ur5_control_msg.acceleration = accel
        self.ur5_control_msg.jointcontrol = True
        goal = [d for d in d_xyz] + [d for d in d_rpy]
        self.ur5_control_msg.values = goal
        
    def servoj_msg(self, goal_joints:list, t:float = 0.0, lookahead:float = 0.5, gain:int = 100) -> None:
        if len(goal_joints) != 6:
            raise ValueError("Please provide all joint angles...")
        self.ur5_control_msg.command = "servoj"
        self.ur5_control_msg.time = t
        self.ur5_control_msg.velocity = 0.0
        self.ur5_control_msg.acceleration = 0.0
        self.ur5_control_msg.jointcontrol = True
        self.ur5_control_msg.values = goal_joints
        self.ur5_control_msg.lookahead = lookahead
        self.ur5_control_msg.gain = gain
        
        
    def stopj_msg(self, accel:float = np.pi/4) -> None:
        self.ur5_control_msg.command = "stopj"
        self.ur5_control_msg.time = 0.0
        self.ur5_control_msg.velocity = 0.0
        self.ur5_control_msg.acceleration = accel
        self.ur5_control_msg.jointcontrol = True
        
    
    def load_json(self, filename:str) -> dict:
        if not os.path.isfile(filename):
            raise ValueError(f"Folling json file does not exist: f{filename}")
        with open(filename, "r") as fh:
            data = json.load(fh)
        return data   
    
    
    def go_home_v2(self, go_home_time=5, open_gripper = True, type=1):
        if type == 1:
            self.movej_msg(goal_joints=HOME_JOINTS_RAD,t=go_home_time)
            print(f"UR5 is going to home position 1 [ASSEMBLING]")
            
        elif type == 2:
            self.movej_msg(goal_joints=HOME_JOINTS_RAD_2,t=go_home_time)
            print(f"UR5 is going to home position 2 [STUCK]")                
            
        self.ur5_joint_publisher.publish(self.ur5_control_msg)
        if open_gripper:
            self.r2fg_msg.position = 0
            self.r2fg_publisher.publish(self.r2fg_msg)
        
        rospy.sleep(go_home_time)       
        print(f"Finished going home")
        
        return

    def go_home(self, go_home_time=5, open_gripper = True):
        while self.go_home_thread_flag:
            go_home_flag = rospy.get_param("/go_home",default=False)        
            go_home_2_flag = rospy.get_param("/go_home_2",default=False)        
            if go_home_flag:
                print(f"UR5 is going to home position 1")
                self.movej_msg(goal_joints=HOME_JOINTS_RAD,t=go_home_time)
                self.ur5_joint_publisher.publish(self.ur5_control_msg)
                if open_gripper:
                    self.r2fg_msg.position = 0
                    self.r2fg_publisher.publish(self.r2fg_msg)
                
                rospy.sleep(go_home_time)       
                rospy.set_param("/go_home",False)
            
            if go_home_2_flag:
                print(f"UR5 is going to home position 2 [STUCK]")                
                self.movej_msg(goal_joints=HOME_JOINTS_RAD_2,t=go_home_time)
                self.ur5_joint_publisher.publish(self.ur5_control_msg)
                if open_gripper:
                    self.r2fg_msg.position = 0
                    self.r2fg_publisher.publish(self.r2fg_msg)
                
                rospy.sleep(go_home_time)
                rospy.set_param("/go_home_2",False)
                
            rospy.sleep(1/30)
        return 
                    
    def exe_struct(self, structure):
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
            Goal_object_to_pick = start_waypoint_[1] 
            print(f"current id: {i}, skip id: {skip_id}, Picking up: {Goal_object_to_pick}")
            if skip_id ==i:
                rospy.set_param("/object_description",f"{object_desc}")
                rospy.set_param("/object_color",f"{object_color}")
                rospy.set_param("/structure",structure)
                rospy.set_param("/object_size",f"210x210")
                rospy.set_param("/instruction",instructions)
                rospy.set_param("/orientation",orientation[0])
                rospy.set_param("/orientation_value",orientation[1])
                rospy.set_param("/objects",objects)
                rospy.set_param("/object_goal_poses",object_goal_poses)
                rospy.set_param("/object_goal_rot",object_goal_rot)
                rospy.set_param("/goal_position",goal_poses[i][0])
                rospy.set_param("/call_llm",True)
                rospy.set_param("/go_home_2",True)
                rospy.sleep(1)
                print(f"waiting till human finish helping robot, {time.process_time()}", end = " ")
                while not rospy.get_param("/human_finished",default=False):
                    # print(".")
                    rospy.sleep(1/10)        
                print("\n")                
                # self.go_home_thread_flag = False
                continue
            for wid, w in enumerate(start_waypoint):
                # print(f"wid: {wid} and start waypoint: {w}")
                if wid == 3:
                    break
                goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions,xyz=w[:3],rpy = w[3:6] )
                self.movej_msg(goal_joints=goal_joints,t=w[6])
                if bool(w[7]):
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
                if bool(w[7]):
                    self.r2fg_msg.position = self.max_r2fg
                else:
                    self.r2fg_msg.position = 50

                self.ur5_joint_publisher.publish(self.ur5_control_msg)
                self.r2fg_publisher.publish(self.r2fg_msg)
                rospy.sleep(w[6])
                
            return 
        
                
        
    def start_exp(self):
        start_exp = rospy.get_param("/start_exp",default=False)
        flag  = True
        if start_exp and not rospy.get_param("/go_home"):
        # if start_exp :
            print(f"started experiment")
            rospy.set_param("/start_exp",False)
            rospy.set_param("/human_finished",False)
            structures = ['K','Z','O']
            # structures = ['K','U','Z','R', 'S','O']
            # structure = structures[self.shapes_ids[self.strcture_id]]
            structure = structures[self.strcture_id]
            # structure = structures[3]
            # structure = 'U'
            # structure = self.shapes[self.strcture_id]
            print(f"input shape: {structure}")
            start_waypoints, end_waypoints, object_name, object_color, skip_id, orientation, instructions, goal_poses = self.get_waypoints(shape=structure)
            objects = []      
            object_goal_poses = [] 
            object_goal_rot   = []    
            for i, waypoint in enumerate(zip(start_waypoints, end_waypoints)):     
                # if i < 3:
                #     continue
                start_waypoint_, end_waypoint = waypoint  
                start_waypoint = start_waypoint_[0]
                objects.append(start_waypoint_[1])
                object_goal_poses.append(goal_poses[i][0])
                object_goal_rot.append(goal_poses[i][1][1])
                Goal_object_to_pick = start_waypoint_[1]
                print(f"current id: {i}, skip id: {skip_id}, Picking up: {Goal_object_to_pick}")
                # rospy.sleep(3)
                if i == skip_id:
                    flag = False
                    start_end_pose = goal_poses[i]
                    rospy.set_param("/object_description",f"{object_name}")
                    rospy.set_param("/object_color",f"{object_color}")
                    rospy.set_param("/structure",structure)
                    rospy.set_param("/object_size",f"210x210")
                    rospy.set_param("/instruction",instructions)
                    rospy.set_param("/orientation",orientation[0])
                    rospy.set_param("/orientation_value",orientation[1])
                    rospy.set_param("/objects",objects)
                    rospy.set_param("/object_goal_poses",object_goal_poses)
                    rospy.set_param("/object_goal_rot",object_goal_rot)
                    rospy.set_param("/goal_position",goal_poses[i][0])
                    rospy.set_param("/call_llm",True)
                    rospy.set_param("/go_home_2",True)
                    rospy.sleep(1)
                    print(f"waiting till human finish helping robot, {time.process_time()}", end = " ")
                    while not rospy.get_param("/human_finished",default=False):
                        # print(".")
                        rospy.sleep(1/10)        
                    print("\n")                
                    # self.go_home_thread_flag = False
                    continue 
                # continue 
                # if not flag:
                #     continue
                    
                for wid, w in enumerate(start_waypoint):
                    # print(f"wid: {wid} and start waypoint: {w}")
                    if wid == 3:
                        break
                    goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions,xyz=w[:3],rpy = w[3:6] )
                    self.movej_msg(goal_joints=goal_joints,t=w[6])
                    if bool(w[7]):
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
                    if bool(w[7]):
                        self.r2fg_msg.position = self.max_r2fg
                    else:
                        self.r2fg_msg.position = 50

                    self.ur5_joint_publisher.publish(self.ur5_control_msg)
                    self.r2fg_publisher.publish(self.r2fg_msg)
                    rospy.sleep(w[6])
                    
            print(f"finished the sub experiment part: {1+ self.strcture_id}")
            
            rospy.set_param("/go_home",True)
            rospy.sleep(1)
            self.strcture_id += 1
            if self.strcture_id == 6:
                print(f"Experiement completed....")
                rospy.set_param("FINISHED",True)
                return
        return 
    
 
    
        
if __name__ == "__main__":
    
    mp = Motion_Planner()
    rospy.set_param("/go_home",True)
    rospy.sleep(1.5)
    mp.go_home_thread.start()
    while not rospy.is_shutdown() and mp.go_home_thread_flag:
        mp.start_exp()
        if mp.strcture_id == 6:
            break
        rospy.sleep(1)
            