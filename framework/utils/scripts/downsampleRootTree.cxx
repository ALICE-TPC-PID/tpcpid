#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TRandom3.h>
#include <thread>
#include <mutex>
#include <vector>

void writeToCSV(const char* rootFileName, const char* csvFileName) {
    // Open the ROOT file
    TFile *file = TFile::Open(rootFileName);

    if (!file || file->IsZombie()) {
        std::cerr << "Error opening ROOT file: " << rootFileName << std::endl;
        return;
    }

    // Open CSV file for writing
    std::ofstream csvFile(csvFileName);

    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file: " << csvFileName << std::endl;
        file->Close();
        return;
    }

    // Loop over all objects in the file
    TIter next(file->GetListOfKeys());
    TKey *key;
    while ((key = dynamic_cast<TKey*>(next()))) {
        // Check if the object is a TTree
        if (key->ReadObj()->IsA() == TTree::Class()) {
            TTree *tree = dynamic_cast<TTree*>(key->ReadObj());

            // Get the number of branches in the TTree
            int numBranches = tree->GetNbranches();

            // Write header with branch names to CSV file
            for (int i = 0; i < numBranches; ++i) {
                TBranch *branch = tree->GetBranch(tree->GetListOfBranches()->At(i)->GetName());
                csvFile << key->GetName() << "_" << branch->GetName();
                if (i < numBranches - 1) {
                    csvFile << ",";
                }
            }
            csvFile << std::endl;

            // Loop over entries and write values to CSV file
            int numEntries = tree->GetEntries();

            for (int entry = 0; entry < numEntries; ++entry) {
                tree->GetEntry(entry);

                for (int i = 0; i < numBranches; ++i) {
                    TBranch *branch = tree->GetBranch(tree->GetListOfBranches()->At(i)->GetName());
                    TLeaf *leaf = branch->GetLeaf(branch->GetName());

                    // Assuming the branches contain simple types (e.g., float, int)
                    if (leaf->IsA() == TLeafF::Class()) {
                        float value = leaf->GetValue();
                        csvFile << value;
                    } else if (leaf->IsA() == TLeafI::Class()) {
                        int value = leaf->GetValue();
                        csvFile << value;
                    }

                    if (i < numBranches - 1) {
                        csvFile << ",";
                    }
                }

                csvFile << std::endl;
            }
        }
    }

    // Close files and clean up
    file->Close();
    csvFile.close();

    std::cout << "Data written to CSV file: " << csvFileName << std::endl;
}

void downsampleTree(const char* inputFileName, const char* outputFileName, double downsamplingFactor) {
    // Open the input ROOT file
    TFile *inputFile = TFile::Open(inputFileName);

    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error opening input ROOT file: " << inputFileName << std::endl;
        return;
    }

    // Retrieve the first key from the list of keys in the file
    TKey *key = dynamic_cast<TKey*>(inputFile->GetListOfKeys()->At(0));

    if (!key) {
        std::cerr << "No keys found in the input ROOT file." << std::endl;
        inputFile->Close();
        return;
    }

    // Get the name of the TTree
    const char* treeName = key->GetName();

    // Assume the TTree is named "tree" if not provided (change accordingly)
    TTree *inputTree = dynamic_cast<TTree*>(inputFile->Get(treeName));

    if (!inputTree) {
        std::cerr << "Error accessing TTree '" << treeName << "' in input ROOT file." << std::endl;
        inputFile->Close();
        return;
    }

    // Create a new ROOT file for output
    TFile *outputFile = new TFile(outputFileName, "RECREATE");

    // Clone the input TTree to the output TTree
    TTree *outputTree = inputTree->CloneTree(0);

    // Get the number of entries in the input TTree
    Long64_t numEntries = inputTree->GetEntries();

    // Seed for the random number generator
    TRandom3 randGen(0);

    // Branch with the varibale that should be cut on
    TBranch *cutBranch = inputTree->GetBranch("fTPCInnerParam");

    // Loop over entries, apply cuts, and randomly downsample
    for (Long64_t entry = 0; entry < numEntries; ++entry) {
        inputTree->GetEntry(entry);

        // Apply the user-defined cut
        double cutValue = cutBranch->GetLeaf("fTPCInnerParam")->GetValue();
        if (randGen.Rndm() < downsamplingFactor || cutValue > 1) {
            outputTree->Fill();
        }
    }

    // Write the output TTree to the output ROOT file
    outputFile->Write();
    outputFile->Close();

    // Clean up
    inputFile->Close();

    std::cout << "Downsampling and applying cuts completed. Result written to: " << outputFileName << std::endl;
}

std::mutex treeWriteMutex;

void processTree(TTree* inputTree, TFile* outputFile, double adjustedDownsamplingFactor, int treeIndex, int numKeys) {
    try {
        std::cout << "Processing tree: " << treeIndex << " / " << numKeys << " (" << inputTree->GetName() << ")" << std::endl;

        // Clone the input TTree to the output TTree
        treeWriteMutex.lock();
        TTree *outputTree = inputTree->CloneTree(0);
        outputTree->SetName(Form("%s_%d", inputTree->GetName(), treeIndex));
        treeWriteMutex.unlock();

        // Get the number of entries in the input TTree
        Long64_t numEntries = inputTree->GetEntries();
        std::cout << "Tree size before downsampling: " << numEntries << " entries" << std::endl;

        // Seed for the random number generator
        TRandom3 randGen(42);

        // Define the chunk size
        const Long64_t chunkSize = 1000000; // Adjust as needed

        // Loop over entries in chunks
        for (Long64_t startEntry = 0; startEntry < numEntries; startEntry += chunkSize) {
            Long64_t endEntry = std::min(startEntry + chunkSize, numEntries);

            for (Long64_t entry = startEntry; entry < endEntry; ++entry) {
                inputTree->GetEntry(entry);

                if (randGen.Rndm() < adjustedDownsamplingFactor) {
                    outputTree->Fill();
                }
            }
        }
        // Print the size of the tree after downsampling
        std::cout << "Tree size after downsampling: " << outputTree->GetEntries() << " entries" << std::endl;

        // Lock the mutex before writing to the output file
        treeWriteMutex.lock();
        outputFile->cd(); // Ensure the output file is the current directory
        outputTree->Write();
        treeWriteMutex.unlock();
    } catch (const std::exception& e) {
        std::cerr << "Error processing tree: " << inputTree->GetName() << ". Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred while processing tree: " << inputTree->GetName() << std::endl;
    }
}

void downsampleFile(const char* inputFileName, const char* outputFileName, int keep_per_tree = 1000000) {
    // Open the input ROOT file
    ROOT::EnableThreadSafety();
    TFile *inputFile = TFile::Open(inputFileName);

    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error opening input ROOT file: " << inputFileName << std::endl;
        return;
    }

    // Get the list of keys in the file
    TList *keyList = inputFile->GetListOfKeys();
    int numKeys = keyList->GetEntries();

    if (numKeys == 0) {
        std::cerr << "No keys found in the input ROOT file." << std::endl;
        inputFile->Close();
        return;
    } else {
        std::cout << "Found trees:" << std::endl;
        TIter next(keyList);
        TKey *key;
        while ((key = dynamic_cast<TKey*>(next()))) {
            if (key->ReadObj()->IsA() == TTree::Class()) {
                std::cout << key->GetName() << "; ";
            }
        }
        std::cout << std::endl;
    }

    // Create a new ROOT file for output
    TFile *outputFile = new TFile(outputFileName, "RECREATE");

    // Vector to hold threads
    std::vector<std::thread> threads;

    // Maximum number of trees to be processed at any time
    const int maxConcurrentTrees = 1;

    // Loop over all keys in the file
    TIter next(keyList);
    int treeIndex = 0;
    for (int i = 0; i < numKeys; ++i) {
        TKey *key = dynamic_cast<TKey*>(next());
        // Check if the object is a TTree
        if (key && key->ReadObj()->IsA() == TTree::Class()) {
            TTree *inputTree = dynamic_cast<TTree*>(key->ReadObj());
            Long64_t treeSize = inputTree->GetTotBytes();
            if (treeSize > 2e11) {  // 20 GB limit
                std::cerr << "Warning: TTree too large (" << treeSize << ") to load, skipping this tree." << std::endl;
            } else {
                // One element is ~64 bytes, so we can estimate the size of the output tree
                float downsamplingFactor = keep_per_tree * 64.f / treeSize;
                threads.emplace_back(processTree, inputTree, outputFile, downsamplingFactor, ++treeIndex, numKeys);

                // If the maximum number of concurrent trees is reached, join the threads
                if (threads.size() >= maxConcurrentTrees) {
                    for (auto& thread : threads) {
                        thread.join();
                    }
                    threads.clear();
                }
            }
        }
    }

    // Join any remaining threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Close the output file
    outputFile->Close();

    // Clean up
    inputFile->Close();

    std::cout << "Downsampling and applying cuts completed. Result written to: " << outputFileName << std::endl;
}


void downsampleRootTree(){
    // Example of using ROOT's built-in method to get options
    TString inputPath = gSystem->Getenv("INPUT_PATH");  // Use environment variables or gSystem to get paths
    TString outputPath = gSystem->Getenv("OUTPUT_PATH");
    int numEvents = 6000000;  // Default value

    // Check if ROOT is running in batch mode (non-interactive)
    if (gROOT->IsBatch()) {
        printf("Running in batch mode\n");
    }
    
    // Make sure paths are provided
    if (inputPath.Length() == 0 || outputPath.Length() == 0) {
        std::cerr << "Please provide input and output paths." << std::endl;
        return;
    }
    // Call downsampleFile with the provided arguments
    downsampleFile(inputPath.Data(), outputPath.Data(), numEvents);   
}
