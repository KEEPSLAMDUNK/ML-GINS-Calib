# Optimization Node (only build if GTSAM is available)
if(GTSAM_FOUND)
    add_executable(ml_gins_calibrator 
        optimization/calibration_node.cpp
        optimization/extrinsic_calibrator.cpp
        optimization/sensor_data_loader.cpp)
    target_compile_options(ml_gins_calibrator PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(ml_gins_calibrator PRIVATE 
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR})
    target_link_libraries(ml_gins_calibrator 
        ${PCL_LIBRARIES} 
        ${OpenMP_CXX_LIBRARIES} 
        Threads::Threads 
        gtsam 
        ${YAML_CPP_LIBRARIES}
        Eigen3::Eigen
        ${CERES_LIBRARIES}
        MANIF::manif)
else()
    message(STATUS "Skipping ml_gins_calibrator target due to missing GTSAM dependency")
endif()

# Evaluation Node (using system manif library)
add_executable(ml_gins_evaluator evaluation/ext_evaluation.cpp)
target_include_directories(ml_gins_evaluator PRIVATE 
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_link_libraries(ml_gins_evaluator 
    ${PCL_LIBRARIES} 
    Eigen3::Eigen
    MANIF::manif)

# Hand-eye calibration library
add_library(ml_gins_hand_eye_calib SHARED 
    hand_eye_calib/HandEyeCalibration.cc 
    hand_eye_calib/PlanarHandEyeCalibration.cc)
target_include_directories(ml_gins_hand_eye_calib PUBLIC 
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_link_libraries(ml_gins_hand_eye_calib 
    ${CERES_LIBRARIES} 
    Eigen3::Eigen)

# Hand-eye calibration test executable
add_executable(ml_gins_hand_eye_test hand_eye_calib/main.cpp)
target_include_directories(ml_gins_hand_eye_test PRIVATE 
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_link_libraries(ml_gins_hand_eye_test 
    ml_gins_hand_eye_calib 
    ${CERES_LIBRARIES}
    Eigen3::Eigen) 