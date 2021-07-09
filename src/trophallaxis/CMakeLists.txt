message("Built trophallaxis")

list(APPEND CMAKE_PREFIX_PATH "../libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})


add_library(trophallaxis SHARED trophallaxis.cpp)
#target_link_libraries(trophallaxis "${TORCH_LIBRARIES}")

#add_executable(trophallaxis trophallaxis.cpp)
target_link_libraries( trophallaxis "${TORCH_LIBRARIES}")
set_property(TARGET trophallaxis PROPERTY CXX_STANDARD 14)