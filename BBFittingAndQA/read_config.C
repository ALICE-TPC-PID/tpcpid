#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// Declare global variables
string Year;
string Period;
int Pass;
string dEdxSelection;
string Tag1;
string Tag2;
string Path;

// Function to read the config file and set global variables
void readConfig() {
    ifstream configFile("config.txt");
    if (!configFile.is_open()) {
        cout << "Error: Unable to open file!" << endl;
        return;
    }

    string line;
    while (getline(configFile, line)) {
        size_t delimiterPos = line.find('=');
        if (delimiterPos != string::npos) {
            string key = line.substr(0, delimiterPos);
            string value = line.substr(delimiterPos + 1);

            // Assign values based on keys
            if (key == "Year") {
                Year = value;
            } else if (key == "Period") {
                Period = value;
            } else if (key == "Pass") {
                Pass = stoi(value); // Convert string to integer
            } else if (key == "dEdxSelection") {
                dEdxSelection = value;
            } else if (key == "Tag1") {
                Tag1 = value;
            } else if (key == "Tag2") {
                Tag2 = value;
            } else if (key == "Path") {
                Path = value;
            }
        }
    }

    configFile.close();
}