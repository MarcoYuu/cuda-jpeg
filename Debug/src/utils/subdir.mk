################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/utils/cuda_timer.cpp \
../src/utils/in_bit_stream.cpp \
../src/utils/out_bit_stream.cpp \
../src/utils/timer.cpp \
../src/utils/util_cv.cpp 

CU_SRCS += \
../src/utils/gpu_in_bit_stream.cu \
../src/utils/gpu_out_bit_stream.cu 

CU_DEPS += \
./src/utils/gpu_in_bit_stream.d \
./src/utils/gpu_out_bit_stream.d 

OBJS += \
./src/utils/cuda_timer.o \
./src/utils/gpu_in_bit_stream.o \
./src/utils/gpu_out_bit_stream.o \
./src/utils/in_bit_stream.o \
./src/utils/out_bit_stream.o \
./src/utils/timer.o \
./src/utils/util_cv.o 

CPP_DEPS += \
./src/utils/cuda_timer.d \
./src/utils/in_bit_stream.d \
./src/utils/out_bit_stream.d \
./src/utils/timer.d \
./src/utils/util_cv.d 


# Each subdirectory must supply rules for building sources it contributes
src/utils/%.o: ../src/utils/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src/utils" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/utils/%.o: ../src/utils/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src/utils" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --device-c -G -I../ -I../~/Library/boost -I../~/Library/ -I../~/NVIDIA_GPU_Computing_SDK/shared/inc -O0 -g -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


