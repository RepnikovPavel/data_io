cmake_minimum_required(VERSION 3.16)
project(train_tokenizer_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# PCRE2 is only needed for mode B (--corpus). Without it the trainer still
# supports mode A (--words words.bin).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PCRE2 QUIET libpcre2-8)
endif()

add_executable(train_tokenizer_cpp
  src/main.cpp
  src/checkpoint.cpp
  src/trainer.cpp
  src/json_writer.cpp
  src/pretok.cpp
)
target_link_libraries(train_tokenizer_cpp PRIVATE Threads::Threads)

if(PCRE2_FOUND)
  target_compile_definitions(train_tokenizer_cpp PRIVATE TOKENIZER_CPP_HAVE_PCRE2)
  target_include_directories(train_tokenizer_cpp PRIVATE ${PCRE2_INCLUDE_DIRS})
  target_link_libraries(train_tokenizer_cpp PRIVATE ${PCRE2_LIBRARIES})
  target_link_directories(train_tokenizer_cpp PRIVATE ${PCRE2_LIBRARY_DIRS})
  message(STATUS "PCRE2 found: corpus mode B enabled")
else()
  message(STATUS "PCRE2 not found: corpus mode B disabled (use --words)")
endif()
