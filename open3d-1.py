import open3d as o3d
import numpy as np

print("->Read point cloud data...")
pcd = o3d.io.read_point_cloud("data/bunny.pcd")
print(pcd)

print("->Visualization point cloud...")
radius =0.01
max_nn = 30
pcd.estimate_normals(search_param =o3d.geometry.KDTreeSearchParamHybrid(radius,max_nn))
print(np.asarray(pcd.normals)[:10, :])
o3d.visualization.draw_geometries([pcd],
                                  window_name ="Show Normal",
                                  width =600,
                                  height =450,
                                  left =30,
                                  top=30,
                                  point_show_normal=True)
