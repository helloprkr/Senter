// Senter CLI - Native C++ Chat Interface with Focus System
// Built on llama.cpp chat infrastructure
// Features: SENTER.md integration, chat history, ASCII branding, colors

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

// llama.cpp includes
#include "common.h"
#include "llama.h"
#include "chat.h"
#include "chat-parser.h"

// ANSI Color codes
#define COLOR_RESET          "\033[0m"
#define COLOR_MINT          "\033[38;5;208m"  // Senter mint (primary)
#define COLOR_PAPYRUS       "\033[38;5;251m"  // Light gray/papyrus
#define COLOR_BURNT_GOLD    "\033[38;5;136m"  // Gold/burnt gold
#define COLOR_BLUE          "\033[34m"
#define COLOR_GREEN         "\033[32m"
#define COLOR_CYAN          "\033[36m"
#define COLOR_YELLOW        "\033[33m"
#define COLOR_RED           "\033[31m"
#define COLOR_BOLD         "\033[1m"

// Chat history structure
struct ChatMessage {
    std::string role;
    std::string content;
    std::string timestamp;
    double relevance_score;
};

struct ChatHistory {
    std::string focus_name;
    std::vector<ChatMessage> recent_messages;  // Last 2
    std::vector<ChatMessage> relevant_exchanges;  // Top 2 by relevance
};

// SENTER.md parser
class SENTER_MD_Parser {
public:
    SENTER_MD_Parser(const std::string& focus_name) 
        : focus_name(focus_name) {}
    
    std::string get_system_prompt() {
        std::ifstream senter_file(get_senter_path());
        if (!senter_file.is_open()) {
            return "You are Senter, a universal AI assistant.";
        }
        
        std::string line;
        bool in_system_prompt = false;
        bool in_yaml = false;
        std::string system_prompt;
        int yaml_depth = 0;
        
        while (std::getline(senter_file, line)) {
            // Track YAML frontmatter
            if (line.find("---") == 0) {
                yaml_depth++;
                in_yaml = (yaml_depth >= 1 && yaml_depth < 2);
                continue;
            }
            
            if (in_yaml && line.find("system_prompt:") != std::string::npos) {
                // Found system_prompt marker, next lines are the prompt
                in_system_prompt = true;
                // Skip the system_prompt: | line itself
                continue;
            }
            
            // Multi-line system prompt (indented content)
            if (in_system_prompt) {
                if (!line.empty() && line[0] != ' ') {
                    // End of multi-line block
                    in_system_prompt = false;
                } else {
                    system_prompt += line + "\n";
                }
            }
        }
        
        // Clean up system prompt (remove indentation)
        std::string clean_prompt;
        std::istringstream iss(system_prompt);
        std::string word;
        while (iss >> word) {
            clean_prompt += word + " ";
        }
        
        return clean_prompt;
    }
    
    std::string get_senter_path() {
        // Try multiple locations
        std::vector<std::string> paths = {
            "./Focuses/" + focus_name + "/SENTER.md",
            "../Focuses/" + focus_name + "/SENTER.md",
            "/home/sovthpaw/ai-toolbox/Senter/Focuses/" + focus_name + "/SENTER.md"
        };
        
        for (const auto& path : paths) {
            std::ifstream test(path);
            if (test.is_open()) {
                return path;
            }
        }
        return "";
    }
    
    std::string get_focus_name() const { return focus_name; }
    
private:
    std::string focus_name;
};

// Chat history manager
class HistoryManager {
public:
    HistoryManager() {}
    
    ChatHistory load_history(const std::string& focus_name) {
        ChatHistory history;
        history.focus_name = focus_name;
        
        SENTER_MD_Parser parser(focus_name);
        std::string senter_path = parser.get_senter_path();
        
        std::ifstream senter_file(senter_path);
        if (!senter_file.is_open()) {
            return history;
        }
        
        std::string line;
        bool in_chat_history = false;
        bool in_recent = false;
        bool in_relevant = false;
        ChatMessage current_msg;
        
        while (std::getline(senter_file, line)) {
            if (line.find("## Chat History") != std::string::npos) {
                in_chat_history = true;
                continue;
            }
            
            if (in_chat_history) {
                if (line.find("### Recent Messages (Last 2)") != std::string::npos) {
                    in_recent = true;
                    in_relevant = false;
                    continue;
                }
                if (line.find("### Relevant Exchanges (Top 2 for this Focus)") != std::string::npos) {
                    in_recent = false;
                    in_relevant = true;
                    continue;
                }
                if (line.find("**[") == 0 || line.find("**[") == 0) {
                    // Parse message: **[YYYY-MM-DD HH:MM:SS]** Role: content
                    if (line.size() > 25) {
                        size_t end_bracket = line.find("]**");
                        if (end_bracket != std::string::npos) {
                            std::string timestamp = line.substr(3, end_bracket - 3);
                            
                            size_t role_colon = line.find(":", end_bracket);
                            if (role_colon != std::string::npos) {
                                std::string role = line.substr(end_bracket + 2, role_colon - end_bracket - 2);
                                std::string content = line.substr(role_colon + 2);
                                
                                current_msg.timestamp = timestamp;
                                current_msg.role = role;
                                current_msg.content = content;
                                current_msg.relevance_score = in_recent ? 1.0 : 0.9;
                                
                                if (in_recent) {
                                    history.recent_messages.push_back(current_msg);
                                } else if (in_relevant) {
                                    history.relevant_exchanges.push_back(current_msg);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return history;
    }
    
    void save_message(const std::string& focus_name, const std::string& role, 
                     const std::string& content, const std::string& timestamp) {
        SENTER_MD_Parser parser(focus_name);
        std::string senter_path = parser.get_senter_path();
        
        // Read entire file
        std::vector<std::string> lines;
        std::ifstream senter_file(senter_path);
        if (!senter_file.is_open()) {
            return;
        }
        
        std::string line;
        while (std::getline(senter_file, line)) {
            lines.push_back(line);
        }
        senter_file.close();
        
        // Check if Chat History section exists
        bool has_chat_history = false;
        for (const auto& l : lines) {
            if (l.find("## Chat History") != std::string::npos) {
                has_chat_history = true;
                break;
            }
        }
        
        // If no Chat History, add it
        if (!has_chat_history) {
            lines.push_back("\n---\n## Chat History\n\n### Recent Messages (Last 2)\n\n");
        } else {
            // Find Recent Messages section
            bool added = false;
            for (size_t i = 0; i < lines.size() && !added; i++) {
                if (lines[i].find("### Recent Messages (Last 2)") != std::string::npos) {
                    // Add new message after this section
                    std::string new_msg = "\n**[" + timestamp + "]** " + role + ": " + content + "\n";
                    lines.insert(lines.begin() + i + 1, new_msg);
                    added = true;
                    
                    // Keep only last 2
                    // Count messages in this section
                    int msg_count = 1;
                    for (size_t j = i + 2; j < lines.size(); j++) {
                        if (lines[j].find("**[") == 0 || lines[j].find("**[") == 0) {
                            msg_count++;
                            if (msg_count > 2) {
                                // Remove old message
                                lines.erase(lines.begin() + j);
                                j--;
                            }
                        } else if (lines[j].find("### Relevant") != std::string::npos) {
                            // End of this section
                            break;
                        }
                    }
                    break;
                }
            }
        }
        
        // Write back
        std::ofstream out_file(senter_path);
        for (const auto& l : lines) {
            out_file << l << "\n";
        }
        out_file.close();
    }
};

// ASCII Art: Senter branding with mint green
void print_senter_banner() {
    std::cout << COLOR_MINT;
    std::cout << R"(
    ███████╗ ████████╗██████╗██╗      █████╗ █████╗   ████████╗███████╗
    ╚══██╔╝╚══██╔══╝██╔══██╗╚██╗     ██╔██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗
       ██║     ██║   ╚██╔╝██║  ╚████╔╝██║  ██╔╝██║   ╚██╗╚██╗   ██║  ██║  ██║  ██║
       ██║     ██║      ██║   ██║   ██║      ██║       ██║       ██║
       ╚███╗   ╚█████╗██║   ██║   ██║      ██║       ██║       ██║       ██║       ██║
        ╚══╝    ╚════╝╚═╝   ╚═╝   ╚═╝      ╚═╝       ╚═╝       ╚═╝
)";
    std::cout << COLOR_RESET;
    std::cout << COLOR_MINT << "  🌟 Universal AI Personal Assistant" << COLOR_RESET << "\n\n";
    std::cout << "    " << COLOR_MINT << "Colors:" << COLOR_RESET << " mint (primary), papyrus/burnt gold (text)\n";
}

void print_usage(const char* prog_name) {
    std::cout << COLOR_CYAN << "Usage:" << COLOR_RESET << "\n";
    std::cout << "  " << prog_name << " -m <model.gguf> -f <focus_name> [options]\n\n";
    std::cout << COLOR_CYAN << "Options:" << COLOR_RESET << "\n";
    std::cout << "  -m, --model <path>    Path to GGUF model (required)\n";
    std::cout << "  -f, --focus <name>    Focus name (default: general)\n";
    std::cout << "  -c, --ctx <tokens>    Context window size (default: 8192)\n";
    std::cout << "  -ngl, --n-gpu-layers <num>  GPU layers (default: -1)\n";
    std::cout << "  -h, --help              Show this help\n\n";
    std::cout << COLOR_CYAN << "Commands:" << COLOR_RESET << "\n";
    std::cout << "  /list              List all available Focuses\n";
    std::cout << "  /focus <name>      Switch to a different Focus\n";
    std::cout << "  /history           Show chat history for current Focus\n";
    std::cout << "  /clear             Clear current chat history\n";
    std::cout << "  /exit              Exit Senter\n\n";
}

std::string get_current_timestamp() {
    time_t now = time(nullptr);
    tm* localtm = localtime(&now);
    
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtm);
    return std::string(buffer);
}

int main(int argc, char** argv) {
    // Parse arguments
    std::string model_path;
    std::string focus_name = "general";
    int n_ctx = 8192;
    int n_gpu_layers = -1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--focus") == 0) {
            if (i + 1 < argc) {
                focus_name = argv[++i];
            }
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx") == 0) {
            if (i + 1 < argc) {
                n_ctx = std::atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--n-gpu-layers") == 0) {
            if (i + 1 < argc) {
                n_gpu_layers = std::atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--list") == 0) {
            // List all Focuses
            std::cout << COLOR_CYAN << "📁 Available Focuses:" << COLOR_RESET << "\n";
            std::vector<std::string> focus_dirs = {
                "Focuses/general",
                "Focuses/coding",
                "Focuses/research",
                "Focuses/creative",
                "Focuses/user_personal",
                "Focuses/internal/Router",
                "Focuses/internal/Goal_Detector",
                "Focuses/internal/Tool_Discovery",
                "Focuses/internal/Context_Gatherer",
                "Focuses/internal/Profiler",
                "Focuses/internal/Planner",
                "Focuses/internal/Chat",
                "Focuses/internal/SENTER_Md_Writer"
            };
            
            for (const auto& dir : focus_dirs) {
                std::ifstream test(dir + "/SENTER.md");
                if (test.is_open()) {
                    std::cout << COLOR_GREEN << "  ✓ " << COLOR_RESET << dir.substr(8) << "\n";
                }
            }
            return 0;
        }
    }
    
    if (model_path.empty()) {
        print_senter_banner();
        std::cout << COLOR_RED << "❌ Error: No model specified!" << COLOR_RESET << "\n";
        std::cout << "Use " << COLOR_MINT << "-m <model.gguf>" << COLOR_RESET << " to specify a model\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize llama.cpp
    print_senter_banner();
    std::cout << COLOR_CYAN << "📦 Loading model: " << COLOR_RESET << model_path << "\n";
    std::cout << COLOR_CYAN << "🎯 Focus: " << COLOR_RESET << focus_name << "\n";
    std::cout << COLOR_CYAN << "🧠 Context: " << COLOR_RESET << n_ctx << " tokens\n";
    std::cout << COLOR_CYAN << "⚡ GPU Layers: " << COLOR_RESET << n_gpu_layers << "\n\n";
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cout << COLOR_RED << "❌ Failed to load model!" << COLOR_RESET << "\n";
        return 1;
    }
    
    std::cout << COLOR_GREEN << "✅ Model loaded successfully!" << COLOR_RESET << "\n\n";
    
    // Initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cout << COLOR_RED << "❌ Failed to create context!" << COLOR_RESET << "\n";
        return 1;
    }
    
    // Load system prompt from SENTER.md
    SENTER_MD_Parser parser(focus_name);
    std::string system_prompt = parser.get_system_prompt();
    
    std::cout << COLOR_MINT << "📝 System Prompt:" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << system_prompt.substr(0, 200) << "..." << COLOR_RESET << "\n\n";
    
    // Initialize chat history
    HistoryManager history_manager;
    ChatHistory history = history_manager.load_history(focus_name);
    
    std::cout << COLOR_CYAN << "💬 Chat History: " << COLOR_RESET << history.recent_messages.size() 
              << " recent + " << history.relevant_exchanges.size() << " relevant exchanges\n\n";
    
    // Chat loop
    std::cout << COLOR_GREEN << "✅ Senter ready!" << COLOR_RESET << "\n";
    std::cout << COLOR_YELLOW << "Type your message or /command (type /help for commands)" << COLOR_RESET << "\n\n";
    
    std::string input;
    std::vector<llama_chat_msg> messages;
    
    // Add history to context (last 2 recent + top 2 relevant)
    for (const auto& msg : history.recent_messages) {
        messages.push_back({msg.role, msg.content});
    }
    for (const auto& msg : history.relevant_exchanges) {
        messages.push_back({msg.role, msg.content});
    }
    
    while (true) {
        // Print prompt with Focus indicator
        std::cout << COLOR_MINT << "[" << focus_name << "] " << COLOR_RESET;
        std::cout << COLOR_BURNT_GOLD << "You: " << COLOR_RESET;
        
        // Read input (simple for now)
        if (!std::getline(std::cin, input)) {
            break;  // EOF
        }
        
        // Check for commands
        if (input.empty()) {
            continue;
        }
        
        if (input[0] == '/') {
            // Handle commands
            if (input == "/help" || input == "/h") {
                print_usage(argv[0]);
            } else if (input == "/list") {
                system("cd /home/sovthpaw/ai-toolbox/Senter && ls Focuses");
            } else if (input == "/clear") {
                messages.clear();
                std::cout << COLOR_PAPYRUS << "✅ Chat history cleared" << COLOR_RESET << "\n";
            } else if (input.substr(0, 6) == "/focus") {
                std::string new_focus = input.substr(7);
                std::cout << COLOR_PAPYRUS << "🔄 Switching to Focus: " << COLOR_RESET << new_focus << "\n";
                
                // Save current context
                for (const auto& msg : messages) {
                    history_manager.save_message(focus_name, msg.role, msg.content, get_current_timestamp());
                }
                
                // Switch focus
                focus_name = new_focus;
                parser = SENTER_MD_Parser(focus_name);
                system_prompt = parser.get_system_prompt();
                history = history_manager.load_history(focus_name);
                messages.clear();
                
                // Reload history
                for (const auto& msg : history.recent_messages) {
                    messages.push_back({msg.role, msg.content});
                }
                for (const auto& msg : history.relevant_exchanges) {
                    messages.push_back({msg.role, msg.content});
                }
                
                std::cout << COLOR_GREEN << "✅ Switched to " << new_focus << COLOR_RESET << "\n";
            } else if (input == "/exit" || input == "/quit") {
                std::cout << COLOR_MINT << "👋 Goodbye!" << COLOR_RESET << "\n";
                break;
            } else {
                std::cout << COLOR_RED << "❌ Unknown command: " << COLOR_RESET << input << "\n";
            }
            continue;
        }
        
        // Add user message
        std::string timestamp = get_current_timestamp();
        messages.push_back({"user", input});
        
        // Generate response (simple for now - just echo back)
        std::string response = "I understand. How can I help you with " + focus_name + " topics?";
        messages.push_back({"assistant", response});
        
        // Save to history
        std::string timestamp = get_current_timestamp();
        history_manager.save_message(focus_name, "user", input, timestamp);
        history_manager.save_message(focus_name, "assistant", response, timestamp);
        
        // Save to history
        history_manager.save_message(focus_name, "user", input, timestamp);
        history_manager.save_message(focus_name, "assistant", response, timestamp);
    }
    
    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    
    return 0;
}
