import os
import sys
import numpy as np
   
class Tasks(object):
    def __init__(self) -> None:
        super(Tasks,self).__init__()
        self.workspace = [1000,500]
        
        
        waypoints_set_A = self.generate_traj(start_xyz=[-0.5, 0.2, -0.045], end_xyz= [ 0.00, 0.4, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        waypoints_set_B = self.generate_traj(start_xyz=[-0.5, 0.4, -0.045], end_xyz= [-0.15, 0.4, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        waypoints_set_C = self.generate_traj(start_xyz=[-0.5, 0.6, -0.045], end_xyz= [ 0.15, 0.4, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        waypoints_set_D = self.generate_traj(start_xyz=[ 0.4, 0.6, -0.045], end_xyz= [ 0.15, 0.6, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        waypoints_set_E = self.generate_traj(start_xyz=[ 0.4, 0.4, -0.045], end_xyz= [-0.15, 0.6, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        waypoints_set_F = self.generate_traj(start_xyz=[ 0.4, 0.2, -0.045], end_xyz= [-0.00, 0.6, -0.03], end_rpy=[-np.pi/2.0,np.pi/2.0,-np.pi/2.0])
        
        self.objects = ["Cylinder", "Wall-E robot", "Long Cuboid", "Long House", "Mobile", "Rocket"]
        self.structures_dict = {"O":[self.goal_traj_O, self.objects[0], 3, ["no change" , 0  ], "from bottom"           ],
                                "U":[self.goal_traj_U, self.objects[1], 3, ["same"      , 0  ], "from left"             ],
                                "S":[self.goal_traj_S, self.objects[2], 3, ["90 deg"    , 90 ], "from bottom"           ],
                                "K":[self.goal_traj_K, self.objects[3], 3, ["45 degrees", 45 ], "slide up from bottom"  ],
                                "R":[self.goal_traj_R, self.objects[4], 4, ["- pi/4"    ,-45 ], "insert from the right" ],
                                "Z":[self.goal_traj_Z, self.objects[5], 3, ["45"        , 45 ], "from bottom"           ]
                        #    "X":[self.goal_traj_X, "Wall-E robot" , 3, ["-45"       ,-45 ], "from right"           ]
                           }
         
        self.waypoints = {1:[waypoints_set_A, self.objects[1] , "red"], 
                          2:[waypoints_set_B, self.objects[5] , "red"], 
                          3:[waypoints_set_C, self.objects[2] , "red"],
                          4:[waypoints_set_D, self.objects[0] , "green"], 
                          5:[waypoints_set_E, self.objects[4] , "green"], 
                          6:[waypoints_set_F, self.objects[3] , "green"]}
        
        return
    
    def get_waypoints(self, shape = "U"):
        struct_info               = self.structures_dict.get(shape)      
        end_waypoints, goal_poses = struct_info[0]()
        object_name               = struct_info[1]
        skip_id                   = struct_info[2]
        orientation               = struct_info[3]
        instructions              = struct_info[4]
        
        print(f"Generating waypoints for shape: {shape}")
        keys = list(self.waypoints.keys())
        values = list(self.waypoints.values())
        obj_init_ = [v[1] for v in values]
        sm_id = obj_init_.index(object_name)
        # print(obj_init_)
        # print(keys)
        # print(sm_id)
        while sm_id != skip_id-1:
            obj_init_ = list(np.roll(obj_init_,1))
            keys = list(np.roll(keys,1))
            sm_id = obj_init_.index(object_name)
        # print(obj_init_)
        # print(keys)
        # print(sm_id)
        
        start_waypoints =[]
        object_color = ""
        for key in keys:
            v = self.waypoints.get(key)
            start_waypoints.append([v[0],v[1]]) 
            if v[1] == object_name:
                object_color = v[2]
        
        return start_waypoints, end_waypoints, object_name, object_color, skip_id-1, orientation, instructions, goal_poses
    



    def generate_traj(self, start_xyz, end_xyz, start_rpy =[0,np.pi/2.0,-np.pi/2.0], end_rpy = [0,np.pi/2.0,-np.pi/2.0], long_time = 4, short_time = 3):
        # waypoint: [x, y, z, ee_pose, time_to_finish_waypoint]
        
        waypoints = list()
        
            
        waypoints.append(start_xyz[:2] +[0.10] + start_rpy + [long_time, 0])
        waypoints.append(start_xyz             + start_rpy + [short_time, 0])
        waypoints.append(start_xyz[:2] +[0.10] + start_rpy + [long_time, 1])
         
        waypoints.append(end_xyz[:2]  +[0.10] + end_rpy + [long_time,  1])
        waypoints.append(end_xyz              + end_rpy + [short_time, 1])
        waypoints.append(end_xyz[:2]  +[0.10] + end_rpy + [long_time,  0])
        return waypoints

    def slid_in(self, xyz, delta = 0.2):
        _xyz = list() 
        if xyz[0]>0:
            _xyz = [xyz[0] + delta] + [xyz[1]] + [0.1]
        else:
            _xyz = [xyz[0] - delta] + [xyz[1]] + [0.1]
        
        _xyz_2 = _xyz[:2] +[0.0]
        return _xyz, _xyz_2
        
    def goal_traj_U(self, long_time=4, short_time=3):
        waypoints_B = list()
        end_xyz_B = [0.1,0.36,-0.046]
        end_rpy_B = [0.0,np.pi/2.0,-np.pi/2.0]
        # waypoints_B.append(self.slid_in(end_xyz_B)[0]+ end_rpy_B + [short_time,  1])
        # waypoints_B.append(self.slid_in(end_xyz_B)[1]+ end_rpy_B + [short_time,  1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        waypoints_C = list()
        end_xyz_C = [-0.1,0.36,-0.045]
        end_rpy_C = [0.0,np.pi/2.0,-np.pi/2.0]
        # waypoints_C.append(self.slid_in(end_xyz_C)[0]+ end_rpy_C + [short_time,  1])
        # waypoints_C.append(self.slid_in(end_xyz_C)[1]+ end_rpy_C + [short_time,  1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [-0.1,0.52,-0.0378]
        end_rpy_D = [0.0,np.pi/2.0,-np.pi/2.0]
        # waypoints_D.append(self.slid_in(end_xyz_D)[0]+ end_rpy_D + [short_time,  1])
        # waypoints_D.append(self.slid_in(end_xyz_D)[1]+ end_rpy_D + [short_time,  1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
        waypoints_E = list()
        end_xyz_E = [0.1,0.52,-0.0345]
        end_rpy_E = [0.0,np.pi/2.0,-np.pi/2.0]
        # waypoints_E.append(self.slid_in(end_xyz_E)[0]+ end_rpy_E + [short_time,  1])
        # waypoints_E.append(self.slid_in(end_xyz_E)[1]+ end_rpy_E + [short_time,  1])
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  1])
        waypoints_E.append(end_xyz_E              + end_rpy_E + [short_time, 1])
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  0])
        
        
        waypoints_F = list()
        end_xyz_F = [0.0,0.63,-0.035]
        end_rpy_F = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        # waypoints_F.append(self.slid_in(end_xyz_F)[0]+ end_rpy_F + [short_time,  1])
        # waypoints_F.append(self.slid_in(end_xyz_F)[1]+ end_rpy_F + [short_time,  1])
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  1])
        waypoints_F.append(end_xyz_F              + end_rpy_F + [short_time, 1])
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  0])
        
        
        
        goal_poses = [[[402, 500 - 330], ["0 degrees" , 0 ], "from bottom"],
                      [[606, 500 - 314], ["0  degrees", 0 ], "from bottom"],
                      [[396, 500 - 164], ["0 degrees" , 0 ], "from left"],
                      [[598, 500 - 158], ["0 degrees" , 0 ], "from bottom"],
                      [[492, 500 - 74 ], ["90 degrees", 90], "from bottom"]]
        waypoints = [waypoints_B, waypoints_C, waypoints_E, waypoints_D, waypoints_F]
        return waypoints, goal_poses
    
    
    def goal_traj_O(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.0,0.28,-0.045]  
        end_rpy_A = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        waypoints_B = list()
        end_xyz_B = [0.12,0.375,-0.04]
        end_rpy_B = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        waypoints_C = list()
        end_xyz_C = [-0.1,0.36,-0.04]
        end_rpy_C = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [-0.1,0.52,-0.045]
        end_rpy_D = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
        waypoints_E = list()
        end_xyz_E = [0.12,0.52,-0.04]
        end_rpy_E = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  1])
        waypoints_E.append(end_xyz_E              + end_rpy_E + [short_time, 1])
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  0])
        
        
        waypoints_F = list()
        end_xyz_F = [0.0,0.61,-0.038]
        end_rpy_F = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  1])
        waypoints_F.append(end_xyz_F              + end_rpy_F + [short_time, 1])
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  0])
        
        
        
        goal_poses = [[[1000 - 516, 500 - 416], ["90 degrees", 90], "from left"],
                      [[1000 - 606, 500 - 314], ["0  degrees", 0 ], "from right"],
                      [[1000 - 402, 500 - 330], ["0 degrees" , 0 ], "from bottom"],
                      [[1000 - 396, 500 - 164], ["0 degrees" , 0 ], "from left"],
                      [[1000 - 598, 500 - 158], ["0 degrees" , 0 ], "from bottom"],
                      [[1000 - 492, 500 - 74 ], ["90 degrees", 90], "from bottom"]]
        
        waypoints = [waypoints_A, waypoints_B, waypoints_C, waypoints_D, waypoints_E, waypoints_F]
        return waypoints, goal_poses
    
    
    def goal_traj_S(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.0,0.28,-0.04]
        end_rpy_A = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        waypoints_B = list()
        end_xyz_B = [0.1,0.36,-0.0385]
        end_rpy_B = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        waypoints_C = list()
        end_xyz_C = [0.0,0.45,-0.0387]
        end_rpy_C = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [-0.12,0.52,-0.04]
        end_rpy_D = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
        waypoints_E = list()
        end_xyz_E = [0.0,0.61,-0.04]
        end_rpy_E = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  1])
        waypoints_E.append(end_xyz_E              + end_rpy_E + [short_time, 1])
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  0])
        
        goal_poses = [[[1000 - 516, 500 -416], ["90 degrees", 90], "from left"],
                      [[1000 - 606, 500 -314], ["0  degrees", 0 ], "from bottom"],
                      [[1000 - 504, 500 -238], ["90 degrees", 90], "from bottom"],
                      [[1000 - 396, 500 -164], ["0 degrees" , 0 ], "from left"],
                      [[1000 - 492, 500 -74 ], ["90 degrees", 90], "from bottom"]]
        
        
        waypoints = [waypoints_A, waypoints_B, waypoints_C, waypoints_D,waypoints_E ]
        
        return waypoints, goal_poses
    
    
    
    def goal_traj_R(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.0,0.30,-0.045]
        end_rpy_A = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        
        waypoints_B = list()
        end_xyz_B = [-0.15,0.60,-0.0345]
        end_rpy_B = [np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        
        
        waypoints_C = list()
        end_xyz_C = [0.1,0.39,-0.0385]
        end_rpy_C = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [-0.1,0.39,-0.04]
        end_rpy_D = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
        waypoints_E = list()
        end_xyz_E = [0.0,0.50,-0.0345]
        end_rpy_E = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  1])
        waypoints_E.append(end_xyz_E              + end_rpy_E + [short_time, 1])
        waypoints_E.append(end_xyz_E[:2]  +[0.10] + end_rpy_E + [long_time,  0])
                      
        waypoints_F = list()
        end_xyz_F = [0.1,0.60,-0.0345]
        end_rpy_F = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  1])
        waypoints_F.append(end_xyz_F              + end_rpy_F + [short_time, 1])
        waypoints_F.append(end_xyz_F[:2]  +[0.10] + end_rpy_F + [long_time,  0])
                     
        
        goal_poses = [[[1000 - 516, 500 - 392], ["90 degrees" , 90 ], "from left"],
                      [[1000 - 602, 500 - 292], ["0  degrees" , 0  ], "from left"],
                      [[1000 - 500, 500 - 188], ["90 degrees" , 90 ], "from left"],
                      [[1000 - 388, 500 - 86 ], ["135 degrees", 135], "slide in from left"],
                      [[1000 - 404, 500 - 296], ["0 degrees"  , 0  ], "from bottom"],
                      [[1000 - 592, 500 - 74 ], ["0 degrees"  , 0  ], "from left"]]
                           
        
        waypoints = [waypoints_A,waypoints_C, waypoints_E, waypoints_B, waypoints_D, waypoints_F]
        
        return waypoints, goal_poses
    
    
    def goal_traj_Z(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.0,0.30,-0.045]
        end_rpy_A = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        
        waypoints_B = list()
        end_xyz_B = [-0.05,0.39,-0.039]
        end_rpy_B = [-np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        waypoints_C = list()
        end_xyz_C = [0.05,0.49,-0.0387]
        end_rpy_C = [-np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [0.0,0.60,-0.035]
        end_rpy_D = [-np.pi/2.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
                       
        
        
        goal_poses = [[[1000 - 516,  500 - 394], ["90 degrees", 90], "from left"],
                      [[1000 - 452,  500 - 298], ["45 degrees", 45], "from bottom"],
                      [[1000 - 548,  500 - 194], ["45 degrees", 90], "from bottom"],
                      [[1000 - 498,  500 - 86 ], ["90 degrees", 90], "from left"]]
        
        waypoints = [waypoints_A, waypoints_B, waypoints_C, waypoints_D]
        return waypoints, goal_poses
    
    
    
    def goal_traj_K(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.0,0.35,-0.039]
        end_rpy_A = [-np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        waypoints_B = list()
        end_xyz_B = [0.1,0.36,-0.052]
        end_rpy_B = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        waypoints_C = list()
        end_xyz_C = [0.1,0.525,-0.045]
        end_rpy_C = [0.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [0.0,0.50,-0.0345]
        end_rpy_D = [np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
        goal_poses = [[[1000 - 608, 500 - 334], ["0  degrees" , 0  ], "from bottom to upward"],
                      [[1000 - 598, 500 - 180], ["0 degrees"  , 0  ], "from bottom"],
                      [[1000 - 504, 500 - 348], ["45 degrees" , 45 ], "from right"],
                      [[1000 - 492, 500 - 182], ["135 degrees", 135], "from left"]]
        
        waypoints  = [waypoints_B, waypoints_C, waypoints_A, waypoints_D]
        return waypoints, goal_poses
    
    
    
    def goal_traj_X(self, long_time=4, short_time=3):
        waypoints_A = list()
        end_xyz_A = [0.07,0.38,-0.0399]
        end_rpy_A = [np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  1])
        waypoints_A.append(end_xyz_A              + end_rpy_A + [short_time, 1])
        waypoints_A.append(end_xyz_A[:2]  +[0.10] + end_rpy_A + [long_time,  0])
        
        
        waypoints_B = list()
        end_xyz_B = [-0.075,0.39,-0.048]
        end_rpy_B = [-np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  1])
        waypoints_B.append(end_xyz_B              + end_rpy_B + [short_time, 1])
        waypoints_B.append(end_xyz_B[:2]  +[0.10] + end_rpy_B + [long_time,  0])
        
        
        waypoints_C = list()
        end_xyz_C = [0.06,0.52,-0.0395]
        end_rpy_C = [-np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  1])
        waypoints_C.append(end_xyz_C              + end_rpy_C + [short_time, 1])
        waypoints_C.append(end_xyz_C[:2]  +[0.10] + end_rpy_C + [long_time,  0])
        
        
        waypoints_D = list()
        end_xyz_D = [-0.085,0.54,-0.035]
        end_rpy_D = [np.pi/4.0,np.pi/2.0,-np.pi/2.0]
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  1])
        waypoints_D.append(end_xyz_D              + end_rpy_D + [short_time, 1])
        waypoints_D.append(end_xyz_D[:2]  +[0.10] + end_rpy_D + [long_time,  0])
        
                       
        
        
        goal_poses = [[[1000 - 428,  500 - 300], [" 45 degrees", 45], "from bottom"],
                      [[1000 - 564,  500 - 158], [" 45 degrees", 45], "from bottom"],
                      [[1000 - 418,  500 - 168], ["-45 degrees", -45], "from right"],
                      [[1000 - 568,  500 - 320], ["-45 degrees", -45], "from left"]
                      ]
        
        waypoints = [waypoints_B, waypoints_C, waypoints_D, waypoints_A]
        return waypoints, goal_poses