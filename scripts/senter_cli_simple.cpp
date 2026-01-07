// Senter CLI - Simple Native Chat Interface
// Reads SENTER.md files dynamically and loads system prompts
// Supports dynamic Focus creation and switching

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

// Simple ANSI colors
#define COLOR_RESET   "\033[0m"
#define COLOR_MINT    "\033[38;5;208m"  // Senter mint (primary)
#define COLOR_PAPYRUS "\033[38;5;251m"  // Papyrus/burnt gold (user text)
#define COLOR_BLUE    "\033[34m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_RED     "\033[31m"

namespace fs = std::filesystem;

// Simple SENTER.md parser
std::string get_system_prompt(const std::string& focus_name) {
    // Try multiple locations
    std::vector<std::string> paths = {
        "./Focuses/" + focus_name + "/SENTER.md",
        "../Focuses/" + focus_name + "/SENTER.md",
    };
    
    for (const auto& path : paths) {
        std::ifstream file(path);
        if (!file.is_open()) continue;
        
        std::string line;
        bool in_system_prompt = false;
        std::string system_prompt;
        bool in_yaml = false;
        int yaml_depth = 0;
        
        while (std::getline(file, line)) {
            // Track YAML frontmatter
            if (line.find("---") == 0) {
                yaml_depth++;
                in_yaml = (yaml_depth >= 1 && yaml_depth < 2);
                continue;
            }
            
            if (in_yaml && line.find("system_prompt:") != std::string::npos) {
                in_system_prompt = true;
                continue;
            }
            
            if (in_system_prompt && !line.empty() && line[0] != ' ') {
                system_prompt += line + "\n";
            }
        }
        
        // Clean up indentation
        std::string clean_prompt;
        std::istringstream iss(system_prompt);
        std::string word;
        while (iss >> word) {
            clean_prompt += word + " ";
        }
        
        if (!clean_prompt.empty()) {
            return "You are Senter, a universal AI assistant.";
        }
        
        return clean_prompt;
}

// Discover all available Focuses dynamically
std::vector<std::string> discover_focuses() {
    std::vector<std::string> focuses;
    
    // Try Focuses directory
    if (fs::exists("Focuses")) {
        for (const auto& entry : fs::directory_iterator("Focuses")) {
            if (entry.is_directory()) {
                if (fs::exists(entry.path() / "SENTER.md")) {
                    focuses.push_back(entry.path().filename().string());
                }
            }
        }
    }
    
    return focuses;
}

// Simple ASCII banner
void print_senter_banner() {
    std::cout << COLOR_MINT;
    std::cout << R"(
   ╔████████╗ ███████╗██████╗██████╗██╗  ███╗██╗
   ██║   ████╗██║  ██║██████╗██████╗██║  ██║  ██║
   ██║   ████╗██║  ██║██████╗██████╗██║  ██║  ██║
   ██║   ████╗██║  ██║██████╗██████╗██║  ██║  ██║
   ╚███╗██████╝██║  ╚══███╗════██╔╝██╔══██╔╝██║
        ╚════╝     ╚════╝  ╚════╝   ╚════╝
)";
    std::cout << COLOR_RESET;
    std::cout << COLOR_MINT << "  🌟 Universal AI Personal Assistant" << COLOR_RESET << "\n\n";
}

void print_usage(const char* prog_name) {
    std::cout << COLOR_CYAN << "Usage:" << COLOR_RESET << "\n";
    std::cout << "  " << prog_name << " -m <model.gguf> [options]\n\n";
    std::cout << COLOR_CYAN << "Options:" << COLOR_RESET << "\n";
    std::cout << "  -m, --model <path>     Path to GGUF model (required)\n";
    std::cout << "  -f, --focus <name>     Focus name (default: general)\n";
    std::cout << "  -c, --ctx <tokens>     Context window size (default: 8192)\n";
    std::cout << "  -ngl, --gpu <num>       GPU layers (default: -1)\n";
    std::cout << "  -l, --list               List all available Focuses\n";
    std::cout << "  -h, --help               Show this help\n";
    std::cout << COLOR_CYAN << "\nCommands:" << COLOR_RESET << "\n";
    std::cout << "  /list              List all Focuses\n";
    std::cout << "  /focus <name>      Switch to a different Focus\n";
    std::cout << "  /create <name>     Create a new dynamic Focus\n";
    std::cout << "  /exit              Exit Senter\n";
}

int main(int argc, char** argv) {
    // Parse arguments
    std::string model_path;
    std::string focus_name = "general";
    int n_ctx = 8192;
    int n_gpu_layers = -1;
    bool list_only = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--focus") == 0) {
            if (i + 1 < argc) focus_name = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx") == 0) {
            if (i + 1 < argc) n_ctx = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--gpu") == 0) {
            if (i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            list_only = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (list_only) {
        print_senter_banner();
        std::vector<std::string> focuses = discover_focuses();
        std::cout << COLOR_CYAN << "📁 Available Focuses:" << COLOR_RESET << "\n";
        for (const auto& focus : focuses) {
            std::cout << COLOR_GREEN << "  ✓ " << COLOR_RESET << focus << "\n";
        }
        std::cout << COLOR_YELLOW << "\n💡 Focuses are dynamic - use /create to make new ones!" << COLOR_RESET << "\n";
        return 0;
    }
    
    if (model_path.empty()) {
        print_senter_banner();
        std::cout << COLOR_RED << "❌ Error: No model specified!" << COLOR_RESET << "\n";
        std::cout << "Use " << COLOR_MINT << "-m <model.gguf>" << COLOR_RESET << " to specify a model\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    print_senter_banner();
    std::cout << COLOR_CYAN << "📦 Loading model: " << COLOR_RESET << model_path << "\n";
    std::cout << COLOR_CYAN << "🎯 Focus: " << COLOR_RESET << focus_name << "\n";
    std::cout << COLOR_CYAN << "🧠 Context: " << COLOR_RESET << n_ctx << " tokens\n";
    
    // Load system prompt from SENTER.md
    std::string system_prompt = get_system_prompt(focus_name);
    
    std::cout << COLOR_MINT << "📝 System Prompt:" << COLOR_RESET << "\n";
    std::cout << COLOR_PAPYRUS << system_prompt.substr(0, 150) << COLOR_RESET << "..." << "\n\n";
    
    std::cout << COLOR_GREEN << "✅ Senter CLI ready!" << COLOR_RESET << "\n";
    std::cout << COLOR_PAPYRUS << "Commands:" << COLOR_RESET << " /list, /focus <name>, /create <name>, /exit\n";
    
    // Simple chat loop
    std::cout << COLOR_PAPYRUS << "\n[" << focus_name << "] You: " << COLOR_RESET;
    
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input.empty()) continue;
        
        if (input[0] == '/') {
            // Handle commands
            if (input == "/list") {
                auto focuses = discover_focuses();
                std::cout << COLOR_CYAN << "\n📁 Available Focuses:" << COLOR_RESET << "\n";
                for (const auto& f : focuses) {
                    std::cout << COLOR_GREEN << "  ✓ " << COLOR_RESET << f << "\n";
                }
            } else if (input == "/exit" || input == "/quit") {
                std::cout << COLOR_MINT << "\n👋 Goodbye!" << COLOR_RESET << "\n";
                break;
            } else if (input.substr(0, 7) == "/create") {
                std::string new_focus = input.substr(8);
                std::cout << COLOR_YELLOW << "\n💡 Creating new Focus: " << new_focus << COLOR_RESET << "\n";
                std::cout << COLOR_YELLOW << "💡 Creating Focus directory: Focuses/" << new_focus << COLOR_RESET << "\n";
                std::cout << COLOR_YELLOW << "💡 Creating SENTER.md with basic template..." << COLOR_RESET << "\n";
                
                fs::create_directories("Focuses/" + new_focus);
                
                std::ofstream senter_file("Focuses/" + new_focus + "/SENTER.md");
                senter_file << R"(---
focus:
  type: conversational
  name: )" << new_focus << R"(
  created: )" << get_timestamp() << R"(

system_prompt: |
  You are )" << new_focus << R"( Focus Agent for Senter.
  Help users with topics related to )" << new_focus << R"(.
  Be friendly and helpful.
  
  Note: This is a dynamically created Focus.
  Configure this Focus by editing Focuses/)" << new_focus << R"(/SENTER.md
---

# Context

## User Preferences
[To be populated by Profiler]

## Goals & Objectives
[To be populated by Goal_Detector]
)";
                
                senter_file.close();
                std::cout << COLOR_GREEN << "✅ Focus created!" << COLOR_RESET << "\n";
                std::cout << COLOR_GREEN << "✅ Switching to new Focus..." << COLOR_RESET << "\n";
                focus_name = new_focus;
            } else {
                std::cout << COLOR_RED << "\n❌ Unknown command: " << COLOR_RESET << input << "\n";
            }
            continue;
        }
        
        // Simple echo for now
        std::cout << COLOR_PAPYRUS << "[" << focus_name << "] " << COLOR_MINT << "Senter: " << COLOR_RESET 
                  << "I understand you want to work with " << focus_name << " topics. ";
        std::cout << "This is the simple CLI version - for full functionality use the Python version.\n";
    }
    
    return 0;
}

std::string get_timestamp() {
    time_t now = time(nullptr);
    tm* localtm = localtime(&now);
    
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", localtm);
    return std::string(buffer);
}
