#pragma once

// Per-frame timing and CSV log shared by float_array_node (cpu), float_array_cuda_node (gpu)
// and float_array_pcl_node (pcl). All write the same columns so one script can compare runs.
//
// All durations are steady_clock milliseconds. A stage a backend does not have stays 0:
//   cpu, pcl: convert_ms (PolygonMesh -> payload)
//   gpu:      upload_ms (host -> device), download_ms (device -> host payload)
//   pcl:      normals_ms
// stamp_sec/stamp_nanosec are header.stamp and match rows of the same frame across runs.
//
// status: ok, not_organized, not_enough_finite, no_plane, empty_mesh, error,
//         empty_cloud (pcl, poisson only).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace meshify
{

/// Timings and result of one processed frame, one CSV row.
struct FrameTimes
{
    uint64_t seq{0};
    int32_t stamp_sec{0};
    uint32_t stamp_nanosec{0};
    double unpack_ms{0.0};
    double upload_ms{0.0};
    double plane_ms{0.0};
    double normals_ms{0.0};
    double mesh_ms{0.0};
    double download_ms{0.0};
    double convert_ms{0.0};
    double publish_ms{0.0};
    double total_ms{0.0};
    std::size_t bytes{0};
    std::size_t triangles{0};
    std::string status{"ok"};

    /// Clear all fields and start frame `next_seq`.
    void reset(uint64_t next_seq)
    {
        *this = FrameTimes{};
        seq = next_seq;
    }
};

/// Milliseconds on steady_clock since `t0`.
inline double elapsedMs(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// CSV header line, including the trailing newline.
inline const char * frameCsvHeader()
{
    return "backend,seq,stamp_sec,stamp_nanosec,"
           "unpack_ms,upload_ms,plane_ms,normals_ms,mesh_ms,download_ms,convert_ms,publish_ms,"
           "total_ms,bytes,triangles,status\n";
}

/// Open `path` for appending and write the header into an empty file.
/// Returns false if the file cannot be opened or already holds a different header,
/// so rows of two column layouts never end up in one file.
inline bool openFrameCsv(std::ofstream & csv, const std::string & path)
{
    {
        std::ifstream existing(path);
        std::string first;
        if (existing && std::getline(existing, first) && !first.empty()) {
            std::string expected = frameCsvHeader();
            expected.pop_back();
            if (first != expected) {
                return false;
            }
        }
    }
    csv.open(path, std::ios::out | std::ios::app);
    if (!csv) {
        return false;
    }
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) {
        csv << frameCsvHeader();
    }
    csv.setf(std::ios::fixed);
    csv.precision(3);
    csv.flush();
    return true;
}

/// Append one row for frame `t`. Does nothing if `csv` is not open.
inline void writeFrameCsv(std::ofstream & csv, const char * backend, const FrameTimes & t)
{
    if (!csv.is_open()) {
        return;
    }
    csv << backend << ','
        << t.seq << ','
        << t.stamp_sec << ','
        << t.stamp_nanosec << ','
        << t.unpack_ms << ','
        << t.upload_ms << ','
        << t.plane_ms << ','
        << t.normals_ms << ','
        << t.mesh_ms << ','
        << t.download_ms << ','
        << t.convert_ms << ','
        << t.publish_ms << ','
        << t.total_ms << ','
        << t.bytes << ','
        << t.triangles << ','
        << t.status << '\n';
    csv.flush();
}

}  // namespace meshify
