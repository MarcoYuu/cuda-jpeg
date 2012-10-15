/*
 * main.cpp
 *
 *  Created on: 2012/09/21
 *      Author: Yuu Momma
 */

void cpu_exec(int argc, char *argv[]);
void gpu_exec(int argc, char *argv[]);

int main(int argc, char *argv[]) {
	cpu_exec(argc, argv);
	gpu_exec(argc, argv);

	return 0;
}

