#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
nlohmann::json CONFIG;

// Function to read the config file and set global variables
void readConfig(std::string config_path="../configuration.json") {
    std::ifstream in(config_path);
    std::cout << "Trying to open " << config_path << "\n";
    if(!in){
        std::cerr << "Cannot open " << config_path << "\n";
        return;
    }
    in >> CONFIG;
}

void writeConfig(std::string config_path="../configuration.json") {
    std::ofstream out(config_path);
    out << CONFIG.dump(4);   // 4 = pretty-print with indentation
}

