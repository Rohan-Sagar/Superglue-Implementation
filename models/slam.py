import cv2
import numpy as np
import open3d as o3d
import time

class PointMap(object):
    def __init__(self):
        self.array = [0, 0, 0]

    def get_points(self, tri_points):
        print("Number of points in get_points:", len(tri_points))
        if len(tri_points) > 0:
            tri_points[:, 1:3] = -tri_points[:, 1:3]
            return tri_points
        else:
            return np.array([])

class Display:
    def __init__(self):
        self.aggr_points = []
        self.is_geometry_added = False
        self.point_cloud = None

    def accumulate_points(self, new_points, transformation_matrix, point_cloud, visualizer): 
        # print("Transformation matrix shape:", transformation_matrix.shape)
        new_points_homogeneous = np.hstack((new_points, np.ones((len(new_points), 1))))
        # print("New points homogeneous shape:", new_points_homogeneous.T.shape)
        transformed_points = (transformation_matrix @ new_points_homogeneous.T).T[:, :3]
        
        valid_transformed_points = ~np.any(np.isnan(transformed_points), axis=1)
        transformed_points = transformed_points[valid_transformed_points]
        self.aggr_points.extend(transformed_points)
        self.point_cloud = point_cloud
        
        point_cloud.points = o3d.utility.Vector3dVector(self.aggr_points)
        
        if not self.is_geometry_added:
            print('Adding geometry')
            visualizer.add_geometry(point_cloud)
            print('\nPoint cloud points while adding', point_cloud)
            self.is_geometry_added = True
        else:
            visualizer.add_geometry(point_cloud)
            print('\nPoint cloud points', point_cloud)

        visualizer.poll_events()
        visualizer.update_renderer()
        # visualizer.run()

    # def create_final_viz(self, visualizer):
    #     self.point_cloud.points = o3d.utility.Vector3dVector(self.aggr_points)
    #     
    #     # if not self.is_geometry_added:
    #     print('Adding geometry')
    #     visualizer.add_geometry(self.point_cloud)
    #     print('\nPoint cloud points while adding', self.point_cloud)
    #         # self.is_geometry_added = True
    #     # else:
    #         # visualizer.update_geometry(point_cloud)
    #         # print('\nPoint cloud points', point_cloud)
    #     visualizer.poll_events()
    #     visualizer.update_renderer()

class SLAMProcessor:
    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.global_point_cloud = np.empty((0, 3))
        self.global_pose = np.eye(4)

    def convert_to_homogenous(self, pts):
        return np.vstack([pts.T, np.ones((1, pts.shape[0]))])

    def triangulate_points(self, P0, P1, pts1, pts2):
        """
        P0, P1: Projection matrices for the two views.
        pts1, pts2: Corresponding points in each view.
        """
        pts1 = pts1[:2, :] / pts1[2, :]
        pts2 = pts2[:2, :] / pts2[2, :]

        points_4d_hom = cv2.triangulatePoints(P0, P1, pts1, pts2)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)[:, 0, :]
        return points_3d

    def process_video(self, last_frame, frame, mkpts0, mkpts1):
        cam_intrinsic = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0, 0, 1]])
 
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, cameraMatrix=cam_intrinsic, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, mkpts0, mkpts1, cameraMatrix=cam_intrinsic)
        
        P0 = np.dot(cam_intrinsic, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P1 = np.dot(cam_intrinsic, np.hstack((R, t)))

        # print("MKPTS SHAPE", mkpts0.shape, mkpts1.shape)
        mkpts0_hom = self.convert_to_homogenous(mkpts0)
        mkpts1_hom = self.convert_to_homogenous(mkpts1)

        # print("MKPTS SHAPE new", mkpts0.shape, mkpts1.shape)

        # local_points = self.triangulate_points(P0, P1, mkpts0, mkpts1)
        local_points = self.triangulate_points(P0, P1, mkpts0_hom, mkpts1_hom)
        # print('LOCAL POINTS', local_points.shape)
        local_points_hom = np.hstack((local_points, np.ones((local_points.shape[0], 1))))
        # print('LOCAL POINTS HOM', local_points_hom.shape)
        global_points_hom = np.dot(self.global_pose, local_points_hom.T).T
        print('GLOBAL POINTS HOM', global_points_hom.shape)
        self.global_point_cloud = np.vstack((self.global_point_cloud, global_points_hom[:, :3]))
        print('global_point_cloud', self.global_point_cloud.shape)
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.squeeze()
        self.global_pose = np.dot(self.global_pose, new_pose)

        return global_points_hom[:, :3], self.global_point_cloud, self.global_pose
