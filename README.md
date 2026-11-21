# ROSPointCloudMeshify

## Knoten

| Knoten | Backend | Verfahren |
|---|---|---|
| `organized_mesh_cpu_node` | CPU (PCL) | RANSAC-Ebene + `OrganizedFastMesh` |
| `organized_mesh_gpu_node` | GPU (CUDA) | dieselbe Pipeline auf der GPU |
| `unorganized_mesh_node` | CPU (PCL) | RANSAC-Ebene + Normalen + Greedy- oder Poisson-Mesh, auch für unorganisierte Clouds |

> `organized_mesh_cpu_node` und `organized_mesh_gpu_node` haben denselben Ablauf, dieselben Parameter und dieselben
Logs. Sie unterscheiden sich nur im Upload zur GPU und darin, dass die CPU für RANSAC und das Entpacken
PCL verwendet.

### Topics

| | Topic | Typ |
|---|---|---|
| Eingang | `/points/xyzrgba` | `sensor_msgs/PointCloud2` |
| Ausgang | `/object_mesh/vertices` | `std_msgs/Float32MultiArray` |

> Ausgabeformat: Dreiecksliste, 7 Floats pro Ecke (`x, y, z, r, g, b, a`, Farben 0–1), je 3 Ecken ein Dreieck.
Clouds ohne Farbe werden grün.

### Parameter

| Parameter | Default | Knoten | Bedeutung |
|---|---|---|---|
| `distance_threshold` | `0.01` | alle | RANSAC-Schwelle in m für Punkte auf der Ebene |
| `output_topic` | `/object_mesh/vertices` | alle | Ausgabe-Topic |
| `mode` | `fast` (unorganized: `greedy`) | alle | organized_mesh: nur `fast`; unorganized_mesh: `greedy` oder `poisson` |
| `enable_logging` | `true` | alle | CSV pro Frame schreiben |
| `csv_path` | `meshify_organized_cpu.csv` / `meshify_organized_gpu.csv` / `meshify_unorganized.csv` | alle | CSV-Datei, relativ zum Startordner |
| `queue_depth` | `100` | alle | Länge der Eingangs-Queue |
| `search_radius` | `0.1` | unorganized_mesh_node | Greedy: Suchradius |
| `max_neighbors` | `150` | unorganized_mesh_node | Greedy: max. Nachbarn |
| `normal_k_search` | `20` | unorganized_mesh_node | Nachbarn für die Normalen |

> Ein unbekannter `mode` beendet den Knoten beim Start mit einer Fehlermeldung.

## Voraussetzungen

- Ubuntu 22.04, ROS 2 Humble
- PCL 1.12 mit `ros-humble-pcl-ros` und `ros-humble-pcl-conversions`
- für den GPU-Knoten: CUDA-Toolkit (getestet mit 11.5) und eine NVIDIA-GPU.
  Bei CUDA < 11.6 wird `g++-10` als Host-Compiler gebraucht (`sudo apt install g++-10`), CMake wählt ihn
  automatisch. Ohne CUDA werden die beiden CPU-Knoten trotzdem gebaut.

## Bauen

```bash
cd ~/ROSPointCloudMeshify
source /opt/ros/humble/setup.bash
./build.sh
```

`build.sh` löscht `build/`, `install/` und `log/` in `meshify_ws` und baut alles neu. Danach:

```bash
source ~/ROSPointCloudMeshify/meshify_ws/install/setup.bash
```

Optional nur für die eigene GPU bauen (schneller):

```bash
cd ~/ROSPointCloudMeshify/meshify_ws
colcon build --packages-select meshify --cmake-args -DMESHIFY_CUDA_ARCHITECTURES=86
```

## Starten

```bash
ros2 run meshify organized_mesh_cpu_node
ros2 run meshify organized_mesh_gpu_node --ros-args -p csv_path:=gpu.csv
ros2 run meshify unorganized_mesh_node --ros-args -p csv_path:=poisson.csv -p mode:=poisson
```

Mit Parametern, z. B. ohne CSV:

```bash
ros2 run meshify organized_mesh_gpu_node --ros-args -p enable_logging:=false
```

## CSV-Log

| Spalte | Inhalt |
|---|---|
| `backend` | `organized_cpu`, `organized_gpu` oder `unorganized` |
| `seq` | laufende Frame-Nummer des Knotens |
| `stamp_sec`, `stamp_nanosec` | `header.stamp` der Cloud, ordnet Zeilen verschiedener Läufe demselben Frame zu |
| `unpack_ms` | PointCloud2 einlesen |
| `upload_ms` | nur GPU: Kopie in den Grafikspeicher |
| `plane_ms` | Filter für gültige Punkte, RANSAC, Ebenenpunkte auf NaN setzen |
| `normals_ms` | nur unorganized_mesh_node: Normalen schätzen |
| `mesh_ms` | Mesh erzeugen |
| `payload_ms` | Mesh → Float-Array im Hauptspeicher (CPU: Umwandlung, GPU: Kopie zurück) |
| `publish_ms` | Nachricht veröffentlichen |
| `total_ms` | ganzer Callback |
| `bytes`, `triangles` | Größe des veröffentlichten Meshs |
| `status` | `ok`, `not_organized`, `not_enough_finite`, `no_plane`, `empty_cloud`, `empty_mesh`, `error` |