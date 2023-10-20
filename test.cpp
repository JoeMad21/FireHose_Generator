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

    std::ofstream myFile("job_script.sh");

    std::string input = "#!/bin/bash\n./gen_demo --device " + std::to_string(device) + " --con_task " + std::to_string(consumption_task) + " --source " + std::to_string(source) + " --dimension " + std::to_string(matrix_dim);

    myFile << input;

    myFile.close();

    system("./job_script.sh");
    
    return 0;
}