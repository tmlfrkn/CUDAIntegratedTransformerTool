#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <vector>
#include <map>

void displayOptions(const std::string& functionName, int count) {
    if (functionName == "__syncthreads") {
        std::cout << "Options for " << functionName << " (" << count << " occurrences):\n";
        std::cout << "1. Replace with NULL\n";
        std::cout << "2. Replace with an empty string\n";
        std::cout << "3. Replace with __syncwarp\n";
        std::cout << "4. Replace with cooperative groups synchronization (TILE4)\n";
        std::cout << "5. Replace with cooperative groups synchronization (ACTIVE)\n";
        std::cout << "6. Pass\n";
    } else if (functionName == "atomicAdd") {
        std::cout << "Options for " << functionName << " (" << count << " occurrences):\n";
        std::cout << "1. Replace with direct operation\n";
        std::cout << "2. Replace with atomicAdd_block\n";
        std::cout << "3. Pass\n";
    } else if (functionName == "if") {
        std::cout << "Options for if statements (" << count << " occurrences):\n";
        std::cout << "1. Simplify function bodies by keeping only the first if statement body\n";
        std::cout << "2. Simplify function bodies by keeping only the else statement body\n";
        std::cout << "3. Simplify function bodies by keeping only the else if statement body\n";
        std::cout << "4. Pass\n";
    } else if (functionName == "double") {
        std::cout << "Options for double variables (" << count << " occurrences):\n";
        std::cout << "1. Convert double variables to float\n";
        std::cout << "2. Pass\n";
    }
}

std::string getCombinedCommand(const std::map<std::string, std::map<int, int>>& choices, const std::string& cudaFilePath, const std::string& includePath = "") {
    std::string command = "CUDAIntegratedTransformerTool ";

    for (const auto& choice : choices) {
        for (const auto& c : choice.second) {
            if (choice.first == "__syncthreads") {
                if (c.second == 1) {
                   if (command.find("--remove_synch_thread_to_null=true") == std::string::npos){
			command += "--remove_synch_thread_to_null=true ";
		    }
		    
                    command += "--syncNULL-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 2) {
                    if (command.find("--remove_synch_thread_to_empty=true") == std::string::npos){
			command += "--remove_synch_thread_to_empty=true ";
		    }
                   
                    command += "--syncEMPTY-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 3) {
		    if (command.find("-replace-with-syncwarp=true ") == std::string::npos) {
			command += "--replace-with-syncwarp=true ";
		    }
                    
                    command += "--syncWARP-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 4) {
		   if (command.find("--synchcooperative=true") == std::string::npos) {
			command += "--synchcooperative=true ";
	    	}
                    
                    command += "--syncCOOP-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 5) {
                    if (command.find("--synchactive=true") == std::string::npos) {
        		command += "--synchactive=true ";
    			}
                    command += "--syncACTIVE-indexes=" + std::to_string(c.first) + " ";
                }
                
            } else if (choice.first == "atomicAdd") {
                if (c.second == 1) {
                  if (command.find("--atomic-to-direct=true") == std::string::npos) {
        		command += "--atomic-to-direct=true ";
    			}
                    command += "--atomicDIRECT-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 2) {
                   if (command.find("--atomic-add-to-atomic-add-block=true") == std::string::npos){
        		command += "--atomic-add-to-atomic-add-block=true ";
    			}
                    command += "--atomicBLOCK-indexes=" + std::to_string(c.first) + " ";
                }
                
            } else if (choice.first == "if") {
                if (c.second == 1) {
                  if (command.find("--simplify-if-statements=true") == std::string::npos){
        		command += "--simplify-if-statements=true ";
    			}
                    command += "--if-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 2) {
                    if (command.find("--simplify-else-statements=true") == std::string::npos){
        		command += "--simplify-else-statements=true ";
    			}
                    
                    
                    
                    command += "--else-indexes=" + std::to_string(c.first) + " ";
                } else if (c.second == 3) {
                    if (command.find("--simplify-else-if-statements=true") == std::string::npos){
        		command += "--simplify-else-if-statements=true ";
    			}
             
            
                    command += "--elseif-indexes=" + std::to_string(c.first) + " ";
                }
               
            } else if (choice.first == "double") {
                if (c.second == 1) {
                  if (command.find("--convert-double-to-float=true") == std::string::npos){
        		command += "--convert-double-to-float=true ";
    			}
                 
                    command += "--double-indexes=" + std::to_string(c.first) + " ";
                }
                
            }
        }
    }

    command += cudaFilePath + " -- --cuda-gpu-arch=sm_86";
    if (!includePath.empty()) {
        command += " -I" + includePath;
    }
    return command;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <CUDA file path> [<include path>]\n";
        return 1;
    }

    std::string cudaFilePath = argv[1];
    std::string includePath = (argc > 2) ? argv[2] : "";

    std::ifstream inFile(cudaFilePath);
    if (!inFile) {
        std::cerr << "Unable to open file " << cudaFilePath << "\n";
        return 1;
    }

    std::string sourceCode((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    std::map<std::string, std::vector<int>> functionOccurrences = {
        {"__syncthreads", {}},
        {"atomicAdd", {}},
        {"if", {}},
        {"double", {}}
    };

    // Analyze the CUDA file for specific functions and types
    std::regex syncRegex("__syncthreads\\s*\\(\\s*\\)");
    std::regex atomicRegex("atomicAdd\\s*\\(.*?\\)");
    std::regex ifRegex("\\bif\\s*\\(.*?\\)");
    std::regex doubleRegex("\\bdouble\\b");

    auto analyzeOccurrences = [&](const std::string& code, const std::regex& regex, const std::string& functionName) {
        auto matches_begin = std::sregex_iterator(code.begin(), code.end(), regex);
        auto matches_end = std::sregex_iterator();
        for (std::sregex_iterator i = matches_begin; i != matches_end; ++i) {
            functionOccurrences[functionName].push_back(0); // Placeholder for user choice
        }
    };

    analyzeOccurrences(sourceCode, syncRegex, "__syncthreads");
    analyzeOccurrences(sourceCode, atomicRegex, "atomicAdd");
    analyzeOccurrences(sourceCode, ifRegex, "if");
    analyzeOccurrences(sourceCode, doubleRegex, "double");

    std::map<std::string, std::map<int, int>> choices; // Updated to store indices and choices

    for (const auto& entry : functionOccurrences) {
        if (!entry.second.empty()) {
            displayOptions(entry.first, entry.second.size());
            for (size_t i = 0; i < entry.second.size(); ++i) {
                std::string choice;
                std::cout << "Enter your choice for occurrence " << (i + 1) << " (or 'pass' to skip): ";
                std::cin >> choice;
                if (choice == "pass") {
                    continue;
                } else {
                    int choiceInt = std::stoi(choice);
                    if ((entry.first == "__syncthreads" && choiceInt >= 1 && choiceInt <= 5) ||
                        (entry.first == "atomicAdd" && choiceInt >= 1 && choiceInt <= 2) ||
                        (entry.first == "if" && choiceInt >= 1 && choiceInt <= 3) ||
                        (entry.first == "double" && choiceInt == 1)) {
                        choices[entry.first][i + 1] = choiceInt;
                    } else {
                        std::cerr << "Invalid choice\n";
                    }
                }
            }
        } else {
            std::cout << "No " << entry.first << "() calls found.\n";
        }
    }

    if (!choices.empty()) {
        std::string command = getCombinedCommand(choices, cudaFilePath, includePath);
        std::cout << "Running command: " << command << "\n";
        std::system(command.c_str());
    }

    return 0;
}

