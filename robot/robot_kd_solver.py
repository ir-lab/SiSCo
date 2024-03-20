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
    """
    Provides forward and inverse kinematics functionalities for a robot using KDL.

    This class initializes with the required data for setting up Kinematics and Dynamics Library (KDL)
    solvers for calculating forward and inverse kinematics. It sets up kinematic chains, and stores
    the solvers as instance attributes to be used by other methods of the class.
    """

    def __init__(self) -> None:
        """
        Initializes a Robot_KD_Solver instance, sets up KDL chain and solvers for the specified robot.

        The initial setup involves defining the base and end effector link names, and performing a one-time
        setup call to configure the kinematic chain and solvers. These names are typically specific to the
        robot's URDF configuration.
        """
        super(Robot_KD_Solver,self).__init__()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ik_solver, self.fk_solver = self.setup_kdl(base_link = "base_link", 
                                                        leaf_link = "fake_end_effector_link")
                                                        

    def setup_kdl(self,base_link,leaf_link, urdf_file = "custom_ur5.urdf") -> tuple:
        """
        One-time setting up of kinematic chain using KDL (Kinematics and Dynamics Library).
        
        Args:
            base_link: The name of the base link of the robotic arm as defined in the URDF model.
            leaf_link: The name of the leaf (end effector) link of the robotic arm as defined in the URDF model.
            urdf_file: The filename of the URDF (Unified Robot Description Format) model file. Defaults
                    to "custom_ur5.urdf" which should represent a modified UR5 robot configuration.
        
        Returns:
            tuple: A tuple containing the initialized inverse kinematics (IK) solver object and the
                forward kinematics (FK) solver object for the specified kinematic chain.
        """
        urdf_path = os.path.join(self.current_dir, "urdf", urdf_file)
        kdl_tree  = kdl_parser_py.urdf.treeFromFile(urdf_path)[1]
        kdl_chain = kdl_tree.getChain(base_link,leaf_link) 
        ik_solver = KDL.ChainIkSolverPos_LMA(kdl_chain, 1e-8 ,1000, 1e-6)
        fk_solver = KDL.ChainFkSolverPos_recursive(kdl_chain)
        return ik_solver, fk_solver
    
    
    def get_ik_sol(self,c_joints,xyz,rpy) -> list:
        """Get inverse kinematic solution given goal xyz (meters), 
           rpy (radian) and current joint angles (radian).
    
        Args:
            c_joints: Current joint angles of the robot arm in radians, typically a list or array.
            xyz: The goal Cartesian coordinates (x, y, z) in meters where the end effector should be positioned.
            rpy: The goal roll, pitch, and yaw in radians describing the end effector's orientation in Euler angles ZYX convention.
        
        Returns:
            list: A list representing the joint angles solution for achieving the
                specified end effector pose. If no solution is found the behavior
                is solver-specific and may return an empty list or raise an exception.
        """
        kdl_c_joints = KDL.JntArray(6)
        for enum, j in enumerate(c_joints):
            kdl_c_joints[enum] = j
        kdl_xyz = KDL.Vector(*xyz)
        kdl_rpy = KDL.Rotation().EulerZYX(*rpy)
        kdl_g_joints = KDL.JntArray(6)
        g_joints = self.ik_solver.CartToJnt(kdl_c_joints, KDL.Frame(kdl_rpy,kdl_xyz), kdl_g_joints)
        return [gj for gj in kdl_g_joints]

    def get_fk_sol(self, joints, segmentNr=-1) -> np.ndarray:
        """Get euler and cartesian frames of end effector 
        from given joint angles.
        Args:
            joints: A list or array-like of joint angles for which forward
                    kinematics will be computed. The number of elements 
                    should match the degrees of freedom of the robot.
            segmentNr: An optional argument that specifies the segment number
                    for which kinematics are to be computed. Default is -1,
                    which often implies the last segment or end effector.
        
        Returns:
            np.ndarray: A numpy array representing the transformation matrix including
                        both the position and orientation of the end effector 
                        in the Cartesian space.
        """

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
    
    def call_home_service(self, home_joints, topic_name, wait_time, robot_type) -> bool:
        """
        Invokes the 'go_home' service in a ROS environment to send a robot to its home position.
        
        Args:
            home_joints: A specification of the joint positions for the robot's home location.
            topic_name: The name of the ROS topic associated with the home service.
            wait_time: The time to wait for the robot to reach its home position.
            robot_type: The type of robot that is being sent home.
        
        Returns:
            bool: The outcome of the service call (True for success, False otherwise).
        
        Raises:
            rospy.ServiceException: If the service call encounters problems.
        """
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
    
    