cmake_minimum_required(VERSION 3.19)

project(PGR)

option(SDL_SHARED "" OFF)
option(SDL_STATIC "" ON)

add_subdirectory(libs/SDL-release-3.2.24)

add_subdirectory(libs/geGL-master)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
	PUBLIC 
	geGL::geGL
	SDL3::SDL3
)