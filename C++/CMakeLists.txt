cmake_minimum_required(VERSION 3.15)
project(EnsembleFind LANGUAGES CUDA CXX)

find_package(CUDA 10 REQUIRED)

set(    CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_35,code=sm_35 )

if( WIN32 )
	set(CMAKE_MSVC_RUNTIME_LIBRARY "")
	set(CMAKE_CXX_FLAGS_RELEASE "/MD")
	set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:LIBCMT")
endif()

cuda_add_executable(EnsembleFind
        DeviceInfo.h
        EnsembleData.h
        kernels.h
        main.cpp
        kernels.cu
        main.h
        Misc.h
        OutputData.h
        OutputMarshaller.h
        OutputRecord.h
        Outputter.h
        ProcessData.h
        ThreadSafeQueue.h)

target_link_libraries(EnsembleFind ${CUDA_LIBRARIES})
