

import numpy as np
import cv2
import os
import sys
import requests
import mixed_reality.intpro_utils as iu
from transforms3d.affines import compose
from transforms3d.euler import mat2euler, euler2mat


class Projections:
    
    def __init__(self, intpro_dir = "/home/slocal/github/intention_projection", proj_x_len = 1.0, proj_y_len = 0.5) -> None:
        # derived from intpro framework
        
        width_hg = int(proj_x_len*1000)
        height_hg = int(proj_y_len*1000)
        # print(width_hg, height_hg)
        # exit()
        self.intrinsic = np.load(os.path.join(intpro_dir, "files", "projector_intrinsic.npy"))
        self.extrinsic = np.load(os.path.join(intpro_dir, "files", "projector_extrinsic.npy"))
        self.distortion = np.array([0.13743797, -0.07864936, -0.05694088, 0.00441552, 0.        ]).reshape(5,1)
        
        # obtained by placing april tag appx at the center of the table
        self.home_tf = np.array([[-0.62948529, 0.77669187, 0.02231616, -0.22428254],
                                 [-0.04680564,-0.06657158, 0.99668323,  1.31073937],
                                 [ 0.77560138, 0.62635291, 0.07825939,  2.04607859],
                                 [ 0.        , 0.        , 0.        ,  1.        ]])
        
        # self.home_tf = self.home_tf @ compose(np.array([0,0,0.05]),euler2mat(0,np.pi/2,0),[1,1,1])
        # self.home_tf = self.home_tf @ compose(np.array([0,0,0.05]),np.eye(3),[1,1,1])
        # get the 3d and its projection on projection frame as 2d plane to project
        self.plane_3d = iu.intpro_utils.get_plane(width = proj_x_len,length = proj_y_len, origin_x = 0, origin_y = 0)
        self.plane_2d = iu.intpro_utils.project3d_2d(K = self.intrinsic, D = self.distortion, tf_mat = self.home_tf, points = self.plane_3d)
        
        self.home_h = self.get_H(points_2d=self.plane_2d, width = width_hg, height = height_hg)
        return 
    
    
    def get_H(self, points_2d, height, width):
        src_mat = np.array([[0     ,  0    ],
                            [height,  0    ],
                            [height,  width],
                            [0     ,  width]],
                            dtype = np.int32)
        H , _ = cv2.findHomography(srcPoints=src_mat, dstPoints= points_2d)
        return H
        
    
    def adjust_and_warp(self, image, rotate_c_90 = False, 
                                      rotate_ac_90 = False, 
                                      rotate_180 = False,
                                      projector_w = 1920, projector_h = 1080):

        if rotate_180:
            image = cv2.rotate(image,cv2.ROTATE_180)
            
        if rotate_c_90:
            image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        
        if rotate_ac_90:
            image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
            
        image = image.astype(np.uint8)
        warped = cv2.warpPerspective(image, self.home_h, (projector_w,projector_h)).astype(np.uint8) 
        return warped 
           

    def show_on_projector(self, image, proj_name = "optoma", x_offxet = 4920, y_offset = 0):
        
        cv2.namedWindow(proj_name,cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(proj_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(proj_name,x_offxet,y_offset)
        cv2.imshow(proj_name,image)
        return 

def init_canvas(h=2500,w=5000):
    return 255*np.ones((h,w,3),dtype=np.uint8)

if __name__ == "__main__":
    
    projection = Projections()
    render_image = cv2.imread("/home/slocal/github/visual_language/generated_vs/23976871418bdf182d93dfd11ec5c0c400e2b7399e83672a1a5be4a81b84814d/output_symbol_3.png")
    # render_image = cv2.resize(render_image,(500,500))
    # render_image = cv2.rotate(render_image, cv2.ROTATE_180)
    canvas = init_canvas()
    projection.home_h = projection.get_H(points_2d=projection.plane_2d, width = canvas.shape[0], height = canvas.shape[1])
    start = [6,0]
    end = [8,5]
    trajectory = np.array([[3,0],
                           [4,0],
                           [4,1],
                           [4,2],
                           [4,3],
                           [4,4],
                           [4,5]
                           ])
    # trajectory = np.array([[0,0],[0,1],[0,2],[1,2],[2,2]])
    # print(trajectory.shape)
    print(render_image.shape)
    for i in range(1,trajectory.shape[0]):
        shift = 10
        xy = trajectory[i-1] * 500
        next_xy = trajectory[i] * 500
        # print(f"move image in grid: {xy} and next_xy: {next_xy}")
        
        if xy[0] < next_xy[0] and xy[1] < next_xy[1]:
            print("first condition met", trajectory[i])
                
        elif xy[0] < next_xy[0] and xy[1] >= next_xy[1]:
            print("second condition met", trajectory[i])
            while xy[0] < next_xy[0]:
                xy[0] += shift
                print(xy[0])
                canvas[xy[0]:500+xy[0],
                       xy[1]:500+xy[1],
                       :] = render_image
                # canvas[:500,:500,:] = render_image
                print(render_image.shape)
                image = projection.adjust_and_warp(canvas,rotate_c_90=True)
                projection.show_on_projector(image)

                cv2.imshow("canvas",cv2.resize(canvas,(500,500)))
                cv2.waitKey(1)
                # canvas = 255*np.ones((5000,5000,3),dtype=np.uint8)
                canvas = init_canvas()
                
        elif xy[0] >= next_xy[0] and xy[1] <next_xy[1]:
            print("third condition met ",trajectory[i] )
            while xy[1] < next_xy[1]:
                xy[1] += shift
                canvas[xy[0]:500+xy[0],
                       xy[1]:500+xy[1],
                       :] = render_image
                # canvas[:500,:500,:] = render_image
                
                image = projection.adjust_and_warp(canvas,rotate_c_90=True)
                projection.show_on_projector(image)

                # cv2.imshow("canvas",cv2.resize(canvas,(500,500)))
                cv2.waitKey(1)
                canvas = init_canvas()


        else:
            print("moving to next grid")
        # h,w,c = canvas.shape
        
        # image = projection.adjust_and_warp(canvas, rotate_c_90=True)
        # projection.show_on_projector(image)
        # cv2.imshow("image",render_image)
        # cv2.imshow("warped",image)
        # cv2.imshow("canvas",cv2.resize(image,(1000,500)))
        
    