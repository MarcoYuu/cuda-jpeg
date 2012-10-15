################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpu_jpeg.cpp \
../src/cpu_main_func.cpp \
../src/main.cpp 

CU_SRCS += \
../src/gpu_jpeg.cu \
../src/gpu_main_func.cu 

CU_DEPS += \
./src/gpu_jpeg.d \
./src/gpu_main_func.d 

OBJS += \
./src/cpu_jpeg.o \
./src/cpu_main_func.o \
./src/gpu_jpeg.o \
./src/gpu_main_func.o \
./src/main.o 

CPP_DEPS += \
./src/cpu_jpeg.d \
./src/cpu_main_func.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --device-c -G -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -O0 -g -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


