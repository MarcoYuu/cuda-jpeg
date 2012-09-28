################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpu_jpeg.cpp \
../src/main.cpp 

CU_SRCS += \
../src/gpu_jpeg.cu \
../src/main_func.cu 

CU_DEPS += \
./src/gpu_jpeg.d \
./src/main_func.d 

OBJS += \
./src/cpu_jpeg.o \
./src/gpu_jpeg.o \
./src/main.o \
./src/main_func.o 

CPP_DEPS += \
./src/cpu_jpeg.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: nvcc compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --device-c -G -O0 -g -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


