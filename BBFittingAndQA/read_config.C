#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
nlohmann::json CONFIG; 

// Function to read the config file and set global variables
void readConfig() {
    std::ifstream in("../Running/configuration.json");
    std::cout << "Trying to open configuration.json\n";
    if(!in){
        std::cerr << "Cannot open configuration.json\n";
        return;
    }
    in >> CONFIG;
}