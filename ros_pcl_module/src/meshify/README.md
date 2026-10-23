# fast_grid_mesh_node

Neuer Knoten im Paket `meshify`. Er vermascht eine **organisierte** Punktwolke (Zivid) in einem Durchlauf und publiziert dasselbe `std_msgs/Float32MultiArray`, das `float_array_unity_plane_node` im Modus `fast` bisher gebaut hat. Die bestehenden Knoten bleiben unverändert.

Pro Frame, ohne die PCL-Mesh-Pipeline:

1. Eine Entpackung der `PointCloud2` in ein kompaktes Punktbild (xyz + Farbe/255).
2. Optionaler Clip an der Unity-Ebene (`/unity/clipping_plane`, lokale +Y, `signed_distance <= distance_threshold` wird verworfen).
3. Gittertriangulierung wie `pcl::OrganizedFastMesh` mit `TRIANGLE_RIGHT_CUT`.
4. Kompaktierung der gültigen Dreiecke direkt in das Wire-Format.

Backends, gleiche Stufenmessung:

| `backend` | Was läuft |
|---|---|
| `cpu` | serieller Referenzpfad |
| `openmp` | dieselben Zellen, OpenMP über das Gitter, geordnete Kompaktierung |
| `cuda` | eigener Kernel, Kompaktierung mit CUB `DeviceSelect::Flagged` |

`cuda` fehlt im Binary, wenn beim Konfigurieren kein CUDA-Toolkit gefunden wurde. Das Paket baut dann mit `cpu` und `openmp`. `backend:=cuda` bricht in dem Fall beim Start mit einer klaren Meldung ab.

## Build

Zielplattform: ROS 2 Humble, Ubuntu 22.04. OpenMP kommt über `find_package(OpenMP)`. CUDA ist optional (`find_package(CUDAToolkit)`). Liegt der Host-GCC über der Toolkit-Grenze (CUDA 12.0–12.3 und GCC 13), nimmt CMake `g++-12`, falls es installiert ist.

Verifiziert wurde der cpu/openmp-Pfad auf Ubuntu 24.04 mit ROS 2 Jazzy (dieselbe OrganizedFastMesh-Semantik wie PCL 1.12 unter Humble). Der CUDA-Code wurde mit Toolkit 12.0 für die Architekturen 75, 80, 86, 89 und 90 übersetzt. Eine GPU war in dieser Umgebung nicht vorhanden, der Kernel ist dort nicht gelaufen.

```bash
cd ros_pcl_module
colcon build --packages-select meshify --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Ohne `-DCMAKE_BUILD_TYPE` setzt dieses Paket **Release** als Default. Ein leerer Build-Typ lässt den numerischen Code unoptimiert; die Stufenzeiten wären dann nicht vergleichbar. `build.sh` übergibt Release ebenfalls.

CUDA abschalten:

```bash
colcon build --packages-select meshify --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DMESHIFY_USE_CUDA=OFF
```

Architekturen (Default je nach Toolkit-Version, z. B. `75;80;86;89;90`):

```bash
colcon build --packages-select meshify --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DMESHIFY_CUDA_ARCHITECTURES=86
```

`nvcc` muss im `PATH` liegen (`/usr/local/cuda/bin` oder das Paket `nvidia-cuda-toolkit`).

## Start

```bash
ros2 launch meshify fast_grid_mesh.launch.py backend:=openmp
```

Oder direkt:

```bash
ros2 run meshify fast_grid_mesh_node --ros-args \
  --params-file $(ros2 pkg prefix meshify)/share/meshify/config/fast_grid_mesh.yaml \
  -p backend:=cpu
```

LOD zur Laufzeit, ohne Neustart:

```bash
ros2 param set /fast_grid_mesh_node triangle_pixel_size 8
ros2 param set /fast_grid_mesh_node grid_stride 4
ros2 param set /fast_grid_mesh_node backend openmp
```

## Parameter

Siehe `config/fast_grid_mesh.yaml`. Die Defaults entsprechen dem bisherigen Fast-Pfad.

| Parameter | Default | Bedeutung |
|---|---|---|
| `backend` | `cpu` | `cpu`, `openmp`, `cuda` |
| `input_topic` | `/points/xyzrgba` | organisierte `PointCloud2` |
| `output_topic` | `/object_mesh/vertices` | `Float32MultiArray` |
| `plane_topic` | `/unity/clipping_plane` | `PoseStamped`, lokale +Y ist die Normale |
| `triangle_pixel_size` | `4` | wie `setTrianglePixelSize(4)` |
| `grid_stride` | `0` | `0` = PCL-Abbildung, sonst Schritt in Pixeln |
| `distance_threshold` | `0.01` | Punkte mit Abstand `<=` Schwelle fallen weg |
| `invert_plane_normal` | `false` | kehrt die Unity-+Y-Normale um |
| `angle_tolerance_deg` | `12.5` | PCL-Schattenkante; `<= 0` schaltet sie aus |
| `max_edge_length_a/b/c` | `0` | Kantenlimit `a + b*d + c*d²`; `0` = aus |
| `use_depth_as_distance` | `false` | wie `useDepthAsDistance` |
| `store_shadowed_faces` | `false` | verworfene Schattendrieiecke behalten |
| `viewpoint_x/y/z` | `0` | PCL-Viewpoint, Default Ursprung |
| `openmp_threads` | `0` | `0` = `omp_get_max_threads()` |
| `log_every_n` | `1` | jeder n-te Frame, der erste immer |

**Schrittweite.** `setTrianglePixelSize(4)` speichert intern `max(1, 4-1) = 3`. Der Knoten macht dasselbe, solange `grid_stride` 0 ist. Ein LOD-Schritt von genau 4 Pixeln ist `grid_stride:=4` (das entspricht `setTrianglePixelSize(5)`).

**Schattenkante.** Der alte Knoten setzt keine maximale Kantenlänge. Aktiv ist nur der PCL-Default: eine Kante, deren Winkel zur Sichtlinie kleiner als 12,5° ist, verwirft das Dreieck. Steile Sprünge (Tischkante zur Kamera) fehlen deshalb auch ohne `max_edge_length`. Ist `max_edge_length_a/b/c` gesetzt, ersetzt das — wie in PCL — den Winkeltest.

**Ebene.** Ohne Pose auf `plane_topic` wird nicht geclippt (Warnung alle 2 s), genau wie im Unity-Plane-Knoten. Die Ebenenrechnung läuft in double.

**QoS.** Eingang: `SensorDataQoS` + `KeepLast(1)` (best effort), damit kein Backlog entsteht. Ausgang: reliable, `KeepLast(1)`, damit der ROS-TCP-Endpoint die Nachricht annimmt. Die Ebenen-Pose ist reliable, `KeepLast(1)`.

## Ausgabeformat

Unverändert gegenüber `convertMeshToFloatArray`:

```
[x, y, z, r, g, b, a] × 3 Vertices × Dreieck
```

`r,g,b` in 0…1, `a = 1`, Reihenfolge der Right-Cut-Dreiecke `(p00, p11, p10)` dann `(p00, p01, p11)`. Kein Header, kein Layout-Feld. Ein leeres Mesh wird nicht publiziert.

Der Mesher schreibt diese 21 Floats pro Dreieck direkt (`TriangleRec` in `include/meshify/mesh_types.hpp`). Ein späteres Kompaktformat kann an dieser Stelle ansetzen, ohne die Punktwolke ein zweites Mal zu kopieren. Es ist nicht Teil dieses Knotens.

## Zeitmessung

Jede verarbeitete Wolke loggt eine Zeile `MESHIFY_TIMING` (steady clock, Mikrosekunden). Dieselben Stufen für alle Backends:

| Feld | Inhalt |
|---|---|
| `deserialize_us` | `PointCloud2` → gepacktes Bild, eine Kopie |
| `upload_us` | Host→GPU, sonst 0 |
| `clip_mesh_us` | Clip + Dreieckstest, schreibt den Slot-Puffer |
| `compact_us` | gültige Dreiecke in den dichten Float-Puffer |
| `download_us` | GPU→Host, sonst 0 |
| `serialize_us` | Float-Puffer in die ROS-Nachricht übernehmen |
| `publish_us` | `publish()` (Middleware-Kopie) |
| `total_us` | Callback gesamt |

`serialize_us` ist klein, weil das Wire-Format schon in `compact` steht. Der erste Frame enthält die Pufferanlage (auf der GPU auch `cudaMalloc`). Den CUDA-Kontext legt der Konstruktor an, nicht der erste Frame.

```bash
ros2 run meshify fast_grid_mesh_node --ros-args -p backend:=cpu
# Log nach backend=openmp und backend=cuda vergleichen, gleiche übrige Parameter.
```

## Tests

```bash
cd ros_pcl_module
colcon test --packages-select meshify --event-handlers console_direct+
colcon test-result --verbose
```

Das GTest `fast_grid_mesh_test` prüft:

- die PCL-Schrittweite (`pixel 4 → stride 3`) gegen die Schleife aus `makeRightCutMesh`,
- die Unity-+Y-Normale,
- das Entpacken von `PointCloud2` (inkl. PCL-`rgb`-Packing und Row-Padding),
- cpu und openmp gegen `pcl::OrganizedFastMesh` (Right Cut, Schattenkante, Ebene, Kantenlimit, `grid_stride`),
- cuda gegen cpu, sobald ein Gerät sichtbar ist. Ohne GPU wird dieser Test übersprungen.

## Auf dem GPU-Rechner

Die Entwicklungs-VM hat keine NVIDIA-GPU. Der CUDA-Pfad ist dort nur kompiliert (nvcc 12.0, Fatbin für sm_75/80/86/89/90). Laufzeit, Treiber und ein voller Zivid-Frame sind dort nicht gemessen. `CudaMatchesCpuWhenDevicePresent` wird ohne Gerät übersprungen.

```bash
nvidia-smi
nvcc --version
cd ros_pcl_module
colcon build --packages-select meshify --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DMESHIFY_CUDA_ARCHITECTURES=86
source install/setup.bash
colcon test --packages-select meshify --event-handlers console_direct+
# Der Test CudaMatchesCpuWhenDevicePresent darf nicht mehr SKIP sein.
ros2 launch meshify fast_grid_mesh.launch.py backend:=cuda
```

`86` ist eine RTX-30-Laptop-GPU. Für eine andere Karte die Compute Capability einsetzen (Turing `75`, Ampere Desktop `80`/`86`, Ada `89`, Hopper `90`). Danach dieselben Szenen mit `backend:=cpu` und `backend:=openmp` laufen lassen und die `MESHIFY_TIMING`-Zeilen vergleichen. Der erste Frame ist der Alloc-Frame.
