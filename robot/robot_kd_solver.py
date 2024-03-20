import numpy as np
import kdl_parser_py.urdf
import PyKDL as KDL
import rospkg
import rospy
from irl_robots.msg import ur5Control, matrix, rows, ur5Joints, gSimpleControl, ur5Tool
from transforms3d.affines import compose
from transforms3d.euler import mat2euler, euler2mat
from transforms3d.quaternions import quat2mat , mat2quat
import os




class Robot_KD_Solver(object):
    """Provides following functionalities
       1. FK 
       2. IK
    """
    def __init__(self) -> None:
        super(Robot_KD_Solver,self).__init__()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ik_solver, self.fk_solver = self.setup_kdl(base_link = "base_link", 
                                                        leaf_link = "fake_end_effector_link")
        # print(self.current_dir)

    def setup_kdl(self,base_link,leaf_link, urdf_file = "custom_ur5.urdf") -> tuple:
        """ one time setting up of kinematic chain using kdl"""
        urdf_path = os.path.join(self.current_dir, "urdf", urdf_file)
        kdl_tree  = kdl_parser_py.urdf.treeFromFile(urdf_path)[1]
        kdl_chain = kdl_tree.getChain(base_link,leaf_link) 
        ik_solver = KDL.ChainIkSolverPos_LMA(kdl_chain, 1e-8 ,1000, 1e-6)
        fk_solver = KDL.ChainFkSolverPos_recursive(kdl_chain)
        return ik_solver, fk_solver
    
    
    def get_ik_sol(self,c_joints,xyz,rpy) -> list:
        """ get inverse kinematic solution given goal xyz (meters), 
            rpy (radian) and current joint angles (radian) """
        kdl_c_joints = KDL.JntArray(6)
        for enum, j in enumerate(c_joints):
            kdl_c_joints[enum] = j
        kdl_xyz = KDL.Vector(*xyz)
        kdl_rpy = KDL.Rotation().EulerZYX(*rpy)
        kdl_g_joints = KDL.JntArray(6)
        g_joints = self.ik_solver.CartToJnt(kdl_c_joints, KDL.Frame(kdl_rpy,kdl_xyz), kdl_g_joints)
        return [gj for gj in kdl_g_joints]

    # @execution_time
    def get_fk_sol(self, joints, segmentNr=-1) -> np.ndarray:
        """ get euler and cartesian frames of end effector 
            from given joint angles"""
        joints_ = KDL.JntArray(6)
        frame   = KDL.Frame()
        for idx,j in enumerate(joints):
            joints_[idx] = j
        self.fk_solver.JntToCart(joints_, frame, segmentNr= segmentNr)
        rot_mat = np.empty((3,3))
        for i in range(3):
            for j in range(3):
                rot_mat[i,j] = frame.M[i,j]
        tf_mat = compose([frame.p[0],frame.p[1],frame.p[2]],rot_mat,[1,1,1])
        return tf_mat
    
    def call_home_service(self, home_joints,  topic_name, wait_time, robot_type) -> bool:
        rospy.wait_for_service("go_home")
        try:
            go_home = rospy.ServiceProxy("go_home",GoHome)
            out = go_home(home_joints,
                          topic_name,
                          wait_time,
                          robot_type)
            return out
        except rospy.ServiceException as e:
            print(f"Go Home Service Call Failed: {e}")
            


           
            
if __name__ == "__main__":
    
    rc = Robot_KD_Solver()
    
    