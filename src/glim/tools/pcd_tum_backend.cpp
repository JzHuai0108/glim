#include <glim/common/cloud_covariance_estimation.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/sub_mapping.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/util/config.hpp>
#include <glim/util/raw_points.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path pcd_dir;
  fs::path tum_path;
  fs::path config_path;
  fs::path out_tum;
  fs::path out_keyframes_tum;
  fs::path keyframe_csv;
  bool export_all_frames = true;
};

struct TumPose {
  std::string stamp_text;
  double stamp = 0.0;
  Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d::Identity();
};

struct PcdEntry {
  fs::path path;
  std::string stamp_text;
  double stamp = 0.0;
};

void print_usage() {
  std::cerr << "Usage: glim_pcd_tum_backend "
            << "--pcd_dir DIR --tum scan_states_odom.txt --config_path CONFIG_DIR --out_tum OUT.tum "
            << "[--out_keyframes_tum OUT_KEYFRAMES.tum] [--keyframe_index_csv keyframe_index_to_input_index.csv] "
            << "[--export_all_frames true|false]" << std::endl;
}

bool parse_bool(const std::string& text) {
  if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF") {
    return false;
  }
  throw std::runtime_error("invalid boolean value: " + text);
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; i++) {
    const std::string key = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--pcd_dir") {
      opts.pcd_dir = require_value(key);
    } else if (key == "--tum") {
      opts.tum_path = require_value(key);
    } else if (key == "--config_path") {
      opts.config_path = require_value(key);
    } else if (key == "--out_tum") {
      opts.out_tum = require_value(key);
    } else if (key == "--out_keyframes_tum") {
      opts.out_keyframes_tum = require_value(key);
    } else if (key == "--keyframe_index_csv") {
      opts.keyframe_csv = require_value(key);
    } else if (key == "--export_all_frames") {
      opts.export_all_frames = parse_bool(require_value(key));
    } else if (key == "--help" || key == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }

  if (opts.pcd_dir.empty() || opts.tum_path.empty() || opts.config_path.empty() || opts.out_tum.empty()) {
    print_usage();
    throw std::runtime_error("missing required arguments");
  }
  if (opts.out_keyframes_tum.empty()) {
    opts.out_keyframes_tum = opts.out_tum;
    opts.out_keyframes_tum.replace_extension(".keyframes.tum");
  }
  if (opts.keyframe_csv.empty()) {
    opts.keyframe_csv = opts.out_tum.parent_path() / "keyframe_index_to_input_index.csv";
  }
  return opts;
}

std::optional<TumPose> parse_tum_line(const std::string& line, int line_number) {
  std::string trimmed = line;
  trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char c) { return !std::isspace(c); }));
  if (trimmed.empty() || trimmed.front() == '#') {
    return std::nullopt;
  }

  std::istringstream iss(trimmed);
  TumPose pose;
  double tx, ty, tz, qx, qy, qz, qw;
  if (!(iss >> pose.stamp_text >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
    std::cerr << "warning: skip malformed TUM line " << line_number << std::endl;
    return std::nullopt;
  }

  pose.stamp = std::stod(pose.stamp_text);
  Eigen::Quaterniond q(qw, qx, qy, qz);
  const double norm = q.norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    std::cerr << "warning: skip TUM line " << line_number << " with invalid quaternion" << std::endl;
    return std::nullopt;
  }
  q.normalize();

  pose.T_world_lidar.setIdentity();
  pose.T_world_lidar.translation() << tx, ty, tz;
  pose.T_world_lidar.linear() = q.toRotationMatrix();
  return pose;
}

std::vector<TumPose> load_tum(const fs::path& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("failed to open TUM file: " + path.string());
  }

  std::vector<TumPose> poses;
  std::string line;
  int line_number = 0;
  while (std::getline(ifs, line)) {
    line_number++;
    auto pose = parse_tum_line(line, line_number);
    if (pose) {
      poses.push_back(*pose);
    }
  }
  std::sort(poses.begin(), poses.end(), [](const TumPose& a, const TumPose& b) { return a.stamp < b.stamp; });
  return poses;
}

std::vector<PcdEntry> load_pcd_list(const fs::path& dir) {
  if (!fs::is_directory(dir)) {
    throw std::runtime_error("PCD directory does not exist: " + dir.string());
  }

  std::vector<PcdEntry> entries;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pcd") {
      continue;
    }
    PcdEntry pcd;
    pcd.path = entry.path();
    pcd.stamp_text = entry.path().stem().string();
    try {
      pcd.stamp = std::stod(pcd.stamp_text);
    } catch (const std::exception&) {
      std::cerr << "warning: skip PCD with non-numeric timestamp stem: " << entry.path() << std::endl;
      continue;
    }
    entries.push_back(pcd);
  }

  std::sort(entries.begin(), entries.end(), [](const PcdEntry& a, const PcdEntry& b) { return a.stamp < b.stamp; });
  return entries;
}

const TumPose* find_pose(
  const PcdEntry& pcd,
  const std::vector<TumPose>& poses,
  const std::unordered_map<std::string, size_t>& exact_pose_indices) {
  const auto exact = exact_pose_indices.find(pcd.stamp_text);
  return exact != exact_pose_indices.end() ? &poses[exact->second] : nullptr;
}

glim::RawPoints::Ptr load_raw_points(const PcdEntry& pcd, double stamp) {
  pcl::PointCloud<pcl::PointXYZI> cloud;
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd.path.string(), cloud) < 0) {
    throw std::runtime_error("failed to load PCD: " + pcd.path.string());
  }

  auto raw = std::make_shared<glim::RawPoints>();
  raw->stamp = stamp;
  raw->points.reserve(cloud.size());
  raw->times.reserve(cloud.size());
  raw->intensities.reserve(cloud.size());

  for (const auto& pt : cloud) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    raw->points.emplace_back(pt.x, pt.y, pt.z, 1.0);
    raw->times.emplace_back(0.0);
    raw->intensities.emplace_back(pt.intensity);
  }
  return raw;
}

glim::EstimationFrame::Ptr create_estimation_frame(
  long id,
  int input_index,
  const TumPose& pose,
  const glim::PreprocessedFrame::ConstPtr& preprocessed,
  const glim::CloudCovarianceEstimation& covariance_estimation) {
  auto frame = std::make_shared<glim::EstimationFrame>();
  frame->id = id;
  frame->stamp = pose.stamp;
  frame->T_lidar_imu.setIdentity();
  frame->T_world_lidar = pose.T_world_lidar;
  frame->T_world_imu = pose.T_world_lidar;
  frame->v_world_imu.setZero();
  frame->imu_bias.setZero();
  frame->raw_frame = preprocessed;
  frame->frame_id = glim::FrameID::LIDAR;

  std::vector<Eigen::Vector4d> normals;
  std::vector<Eigen::Matrix4d> covs;
  covariance_estimation.estimate(preprocessed->points, preprocessed->neighbors, preprocessed->k_neighbors, normals, covs);

  auto cloud = std::make_shared<gtsam_points::PointCloudCPU>(preprocessed->points);
  if (!preprocessed->intensities.empty()) {
    cloud->add_intensities(preprocessed->intensities);
  }
  cloud->add_normals(normals);
  cloud->add_covs(covs);
  frame->frame = cloud;

  frame->custom_data["input_index"] = std::shared_ptr<void>(new int(input_index), [](void* p) { delete static_cast<int*>(p); });
  frame->custom_data["stamp_text"] = std::shared_ptr<void>(new std::string(pose.stamp_text), [](void* p) { delete static_cast<std::string*>(p); });
  return frame;
}

std::string format_stamp(double stamp) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(9) << stamp;
  return oss.str();
}

std::string stamp_text_of(const glim::EstimationFrame::ConstPtr& frame) {
  const std::string* stamp_text = frame->get_custom_data<std::string>("stamp_text");
  return stamp_text ? *stamp_text : format_stamp(frame->stamp);
}

void write_tum_frame(std::ofstream& ofs, const std::string& stamp_text, const Eigen::Isometry3d& pose) {
  Eigen::Quaterniond q(pose.linear());
  q.normalize();
  const Eigen::Vector3d t = pose.translation();
  ofs << stamp_text << " " << std::fixed << std::setprecision(6) << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " " << q.y()
      << " " << q.z() << " " << q.w() << "\n";
}

void ensure_parent_directory(const fs::path& path) {
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

int input_index_of(const glim::EstimationFrame::ConstPtr& frame) {
  const int* input_index = frame->get_custom_data<int>("input_index");
  return input_index ? *input_index : static_cast<int>(frame->id);
}

void write_submap_origin_outputs(const std::vector<glim::SubMap::Ptr>& submaps, const fs::path& tum_path, const fs::path& csv_path) {
  ensure_parent_directory(tum_path);
  ensure_parent_directory(csv_path);
  std::ofstream tum(tum_path);
  if (!tum) {
    throw std::runtime_error("failed to open output keyframe TUM: " + tum_path.string());
  }
  std::ofstream csv(csv_path);
  if (!csv) {
    throw std::runtime_error("failed to open keyframe CSV: " + csv_path.string());
  }

  csv << "submap_index,input_index,timestamp\n";
  for (size_t i = 0; i < submaps.size(); i++) {
    const auto& submap = submaps[i];
    if (!submap || submap->frames.empty()) {
      continue;
    }
    const auto origin = submap->origin_frame();
    const std::string stamp_text = stamp_text_of(origin);
    write_tum_frame(tum, stamp_text, submap->T_world_origin);
    csv << i << "," << input_index_of(origin) << "," << stamp_text << "\n";
  }
}

void write_all_frame_outputs(const std::vector<glim::SubMap::Ptr>& submaps, const fs::path& tum_path) {
  ensure_parent_directory(tum_path);
  std::ofstream tum(tum_path);
  if (!tum) {
    throw std::runtime_error("failed to open output TUM: " + tum_path.string());
  }

  for (const auto& submap : submaps) {
    if (!submap || submap->frames.empty()) {
      continue;
    }

    const Eigen::Isometry3d T_world_endpoint_L = submap->T_world_origin * submap->T_origin_endpoint_L;
    const Eigen::Isometry3d T_odom_lidar0 = submap->frames.front()->T_world_lidar;
    for (const auto& frame : submap->frames) {
      const Eigen::Isometry3d T_world_lidar = T_world_endpoint_L * T_odom_lidar0.inverse() * frame->T_world_lidar;
      write_tum_frame(tum, stamp_text_of(frame), T_world_lidar);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_args(argc, argv);
    glim::GlobalConfig::instance(opts.config_path.string(), true);

    const auto pcds = load_pcd_list(opts.pcd_dir);
    const auto poses = load_tum(opts.tum_path);
    std::cout << "loaded scans: " << pcds.size() << std::endl;
    std::cout << "loaded poses: " << poses.size() << std::endl;

    std::unordered_map<std::string, size_t> exact_pose_indices;
    exact_pose_indices.reserve(poses.size());
    for (size_t i = 0; i < poses.size(); i++) {
      exact_pose_indices.emplace(poses[i].stamp_text, i);
    }

    std::unordered_map<std::string, bool> pcd_stamps;
    pcd_stamps.reserve(pcds.size());
    for (const auto& pcd : pcds) {
      pcd_stamps.emplace(pcd.stamp_text, true);
    }
    int missing_reported = 0;
    for (const auto& pose : poses) {
      const fs::path expected = opts.pcd_dir / (pose.stamp_text + ".pcd");
      if (!pcd_stamps.count(pose.stamp_text) && !fs::exists(expected)) {
        if (missing_reported < 20) {
          std::cerr << "warning: pose has no matching PCD file: " << expected << std::endl;
        }
        missing_reported++;
      }
    }
    if (missing_reported > 20) {
      std::cerr << "warning: " << (missing_reported - 20) << " additional missing PCD paths were omitted" << std::endl;
    }

    glim::CloudPreprocessor preprocessor;
    glim::CloudCovarianceEstimation covariance_estimation;

    glim::SubMappingParams sub_mapping_params;
    sub_mapping_params.enable_imu = false;
    glim::SubMapping sub_mapping(sub_mapping_params);

    glim::GlobalMappingParams global_mapping_params;
    global_mapping_params.enable_imu = false;
    glim::GlobalMapping global_mapping(global_mapping_params);

    std::vector<glim::SubMap::Ptr> optimized_submaps;
    const int callback_id = glim::GlobalMappingCallbacks::on_update_submaps.add([&](const std::vector<glim::SubMap::Ptr>& submaps) { optimized_submaps = submaps; });

    int inserted = 0;
    int skipped_without_pose = 0;
    glim::EstimationFrame::Ptr last_inserted_frame;
    for (size_t i = 0; i < pcds.size(); i++) {
      const TumPose* pose = find_pose(pcds[i], poses, exact_pose_indices);
      if (!pose) {
        skipped_without_pose++;
        continue;
      }

      auto raw = load_raw_points(pcds[i], pose->stamp);
      if (raw->points.empty()) {
        std::cerr << "warning: skip empty PCD after finite-point filtering: " << pcds[i].path << std::endl;
        continue;
      }

      auto preprocessed = preprocessor.preprocess(raw);
      if (!preprocessed || preprocessed->points.empty()) {
        std::cerr << "warning: skip PCD with no preprocessed points: " << pcds[i].path << std::endl;
        continue;
      }

      auto frame = create_estimation_frame(static_cast<long>(i), static_cast<int>(i), *pose, preprocessed, covariance_estimation);
      sub_mapping.insert_frame(frame);
      last_inserted_frame = frame;
      for (const auto& submap : sub_mapping.get_submaps()) {
        global_mapping.insert_submap(submap);
      }
      inserted++;
    }

    if (last_inserted_frame) {
      sub_mapping.insert_frame(last_inserted_frame);
      for (const auto& submap : sub_mapping.get_submaps()) {
        global_mapping.insert_submap(submap);
      }
    }

    for (const auto& submap : sub_mapping.submit_end_of_sequence()) {
      global_mapping.insert_submap(submap);
    }
    global_mapping.optimize();
    glim::GlobalMappingCallbacks::on_update_submaps.remove(callback_id);

    std::cout << "inserted frames: " << inserted << std::endl;
    if (skipped_without_pose) {
      std::cerr << "warning: skipped PCD scans without matching TUM poses: " << skipped_without_pose << std::endl;
    }
    std::cout << "optimized submaps: " << optimized_submaps.size() << std::endl;

    if (optimized_submaps.empty()) {
      throw std::runtime_error("backend produced no submaps");
    }

    write_submap_origin_outputs(optimized_submaps, opts.out_keyframes_tum, opts.keyframe_csv);
    if (opts.export_all_frames) {
      write_all_frame_outputs(optimized_submaps, opts.out_tum);
      std::cout << "saved every-frame optimized poses: " << opts.out_tum << std::endl;
      std::cout << "saved optimized submap poses: " << opts.out_keyframes_tum << std::endl;
    } else {
      write_submap_origin_outputs(optimized_submaps, opts.out_tum, opts.keyframe_csv);
      std::cerr << "warning: only optimized submap/keyframe poses were exported to out_tum" << std::endl;
    }
    std::cout << "saved keyframe mapping CSV: " << opts.keyframe_csv << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
