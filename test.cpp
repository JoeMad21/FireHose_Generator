#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

int main() {

    int device = -1;
    std::cout << "Welcome to the FireHose Generator. What device are you targeting today?" << std::endl;
    std::cout << "1. Graphcore IPU" << std::endl;
    std::cout << "2. UPMEM DPU" << std::endl;
    std::cin >> device;

    int consumption_task = -1;
    std::cout << "What consumption task would you like to do on the back-end?" << std::endl;
    std::cout << "1. Matrix multiplication" << std::endl;
    std::cout << "2. Graph task" << std::endl;
    std::cout << "3. Hashing" << std::endl;
    std::cin >> consumption_task;

    int source = -1;
    std::cout << "Where should we source the data?" << std::endl;
    std::cout << "1. Random Generation" << std::endl;
    std::cout << "2. From file" << std::endl;
    std::cin >> source;

    long unsigned int matrix_dim = 0;
    std::cout << "What dimensions would you like for your square matrix? (NxN)" << std::endl;
    std::cin >> matrix_dim;

    std::ofstream myFile("gen.batch");

    std::string input = "#!/bin/bash\n#SBATCH --job-name FireHose_Generator\n#SBATCH --ipus=1\n--partition=p64\n#SBATCH --nodelist=gc-poplar-03\n#SBATCH --ntasks 1\n#SBATCH --time=00:05:00\n\nsrun ./gen_demo --device " + std::to_string(device) + " --con_task " + std::to_string(consumption_task) + " --source " + std::to_string(source) + " --dimension " + std::to_string(matrix_dim);

    myFile << input;

    myFile.close();

    system("sbatch gen.batch");
    
    return 0;
}