cmake_minimum_required(VERSION 3.16)
project(train_tokenizer_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
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
  pkg_check_modules(UTF8PROC QUIET libutf8proc)
endif()

add_executable(train_tokenizer_cpp
  src/main.cpp
  src/checkpoint.cpp
  src/trainer.cpp
  src/json_writer.cpp
  src/pretok.cpp
)
target_link_libraries(train_tokenizer_cpp PRIVATE Threads::Threads)

# Static libstdc++/libgcc: the bookworm-slim runtime ships an older libstdc++
# than gcc-13 emits references to.
target_link_options(train_tokenizer_cpp PRIVATE -static-libstdc++ -static-libgcc)

if(PCRE2_FOUND)
  target_compile_definitions(train_tokenizer_cpp PRIVATE TOKENIZER_CPP_HAVE_PCRE2)
  target_include_directories(train_tokenizer_cpp PRIVATE ${PCRE2_INCLUDE_DIRS})
  target_link_libraries(train_tokenizer_cpp PRIVATE ${PCRE2_LIBRARIES})
  target_link_directories(train_tokenizer_cpp PRIVATE ${PCRE2_LIBRARY_DIRS})
  message(STATUS "PCRE2 found: corpus mode B enabled")
else()
  message(STATUS "PCRE2 not found: corpus mode B disabled (use --words)")
endif()

# count_tokens requires PCRE2 and utf8proc (mandatory here).
if(NOT PCRE2_FOUND OR NOT UTF8PROC_FOUND)
  message(FATAL_ERROR "count_tokens requires libpcre2-8 and libutf8proc (install libpcre2-dev libutf8proc-dev)")
endif()
add_executable(count_tokens
  src/count_tokens.cpp
  src/encoder.cpp
)
target_link_libraries(count_tokens PRIVATE Threads::Threads ${PCRE2_LIBRARIES} ${UTF8PROC_LIBRARIES})
target_include_directories(count_tokens PRIVATE ${PCRE2_INCLUDE_DIRS} ${UTF8PROC_INCLUDE_DIRS})
target_link_directories(count_tokens PRIVATE ${PCRE2_LIBRARY_DIRS} ${UTF8PROC_LIBRARY_DIRS})
target_link_options(count_tokens PRIVATE -static-libstdc++ -static-libgcc)
