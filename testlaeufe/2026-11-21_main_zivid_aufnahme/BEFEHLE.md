# Knotentest 2026-11-21: Branch `node-test-2026-11-21` (von `main`, 63f502d), Aufnahme `zivid_aufnahme`

## 1. Branch anlegen

```bash
cd ~/ROSPointCloudMeshify
git switch -c node-test-2026-11-21 main
```

## 2. Bauen

```bash
cd ~/ROSPointCloudMeshify
source /opt/ros/humble/setup.bash
./build.sh
```

## 3. Testläufe

Jeder Lauf: Knoten starten, 3 s warten, Aufnahme abspielen, warten bis 46 Zeilen in der CSV stehen, Knoten mit Strg+C (SIGINT) beenden. Automatisiert in `run_test.sh`.

In jedem Terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ROSPointCloudMeshify/meshify_ws/install/setup.bash
export ROS_DOMAIN_ID=42
```

Terminal 1 (je Lauf einer davon):

```bash
mkdir -p ~/testlauf/organized_cpu && cd ~/testlauf/organized_cpu
~/ROSPointCloudMeshify/meshify_ws/install/meshify/lib/meshify/organized_mesh_cpu_node --ros-args -p csv_path:=organized_cpu.csv
```

```bash
mkdir -p ~/testlauf/organized_gpu && cd ~/testlauf/organized_gpu
~/ROSPointCloudMeshify/meshify_ws/install/meshify/lib/meshify/organized_mesh_gpu_node --ros-args -p csv_path:=organized_gpu.csv
```

```bash
mkdir -p ~/testlauf/unorganized_greedy && cd ~/testlauf/unorganized_greedy
~/ROSPointCloudMeshify/meshify_ws/install/meshify/lib/meshify/unorganized_mesh_node --ros-args -p csv_path:=unorganized_greedy.csv -p mode:=greedy
```

```bash
mkdir -p ~/testlauf/unorganized_poisson && cd ~/testlauf/unorganized_poisson
~/ROSPointCloudMeshify/meshify_ws/install/meshify/lib/meshify/unorganized_mesh_node --ros-args -p csv_path:=unorganized_poisson.csv -p mode:=poisson
```

Terminal 2:

```bash
ros2 bag play --wait-for-all-acked 10000 ~/Schreibtisch/zivid_bags/zivid_aufnahme
```

`--wait-for-all-acked 10000` lässt den Player bis zu 10 s warten, bis der Knoten alle Clouds bestätigt hat. Ohne die Option fehlte bei GPU und Greedy im ersten Durchgang die letzte Cloud (45/46), weil der Player sich vorher beendet hat. Diese beiden Läufe wurden mit der Option wiederholt (46/46). CPU und Poisson hatten schon ohne die Option 46/46.

## 4. Ergebnisse (46/46 Frames, alle `ok`, Median in ms)

| Lauf | total | min | max | unpack | upload | plane | normals | mesh | payload | publish | Dreiecke |
|---|---|---|---|---|---|---|---|---|---|---|---|
| organized_cpu | 678.3 | 668.4 | 830.9 | 168.5 | – | 222.8 | – | 98.1 | 134.7 | 0.4 | 57 046 |
| organized_gpu | 35.8 | 34.6 | 85.8 | 10.9 | 19.5 | 2.6 | – | 0.4 | 1.5 | 0.4 | 57 054 |
| unorganized greedy | 6087.4 | 5989.4 | 8297.2 | 154.2 | – | 211.3 | 1185.9 | 4193.8 | 200.2 | 3.6 | 565 914 |
| unorganized poisson | 2305.9 | 2240.4 | 2994.4 | 169.8 | – | 258.0 | 881.8 | 969.0 | 3.5 | 0.2 | 32 589 |

GPU ist ~19× schneller als CPU, bei praktisch gleicher Dreieckszahl. Der Max-Wert ist jeweils der erste Frame (Aufwärmen).
Ergebnisse decken sich mit dem Test vom 2026-11-20.

Rohdaten: `<lauf>/<lauf>.csv` und `<lauf>/node.log`.
