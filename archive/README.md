# Archive

Code that is no longer built or maintained. Kept for reference only.
`COLCON_IGNORE` keeps colcon from picking up anything in here.

- `object_detection/`: predecessor package of `meshify`, incl. its old `backups/`.
- `meshify_nodes/marker_array_node.cpp`, `marker_array_node_2.cpp`: MarkerArray output
  (used by the older ROSUnityBIRPMeshViz project).
- `meshify_nodes/float_array_unity_plane_node.cpp`: Float32MultiArray mesh clipped by a plane
  from Unity (`/unity/clipping_plane`, ROSUnityURPMeshViz `ROSPlanePublisher`).

To bring a node back, move it to `meshify_ws/src/meshify/src/` and add it to `NODES`
in `meshify_ws/src/meshify/CMakeLists.txt` (plus its message dependencies).
