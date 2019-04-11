# NearestNeighbor Depth Completion

Use Voronoi diagram/nearest neighbor to complete a sparse depth image. Used OpenGL to achieve better performance. This can be used as the most simple basline to compare with. We use KITTI dataset.

### Demo

Original sparse depth image

![original_sparse_depth](https://raw.githubusercontent.com/jiangwei221/voronoi_depth_completion/master/readme_materials/images/original_sparse_depth.png)

Original RGB

![original_rgb](https://raw.githubusercontent.com/jiangwei221/voronoi_depth_completion/master/readme_materials/images/original_rgb.png)

NN depth completion(No use of RGB)

![voronoi_completion](https://raw.githubusercontent.com/jiangwei221/voronoi_depth_completion/master/readme_materials/images/voronoi_completion.png)

Confidence map

![completion_confidence_map](https://raw.githubusercontent.com/jiangwei221/voronoi_depth_completion/master/readme_materials/images/completion_confidence_map.png)

Annotated ground truth

![annotated_gt](https://raw.githubusercontent.com/jiangwei221/voronoi_depth_completion/master/readme_materials/images/annotated_gt.png)

### Metrics

| Method        | MAE           | RMSE  |
| ------------- |:-------------:| -----:|
| NN completion      | 434 | 2225 |
