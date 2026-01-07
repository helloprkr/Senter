---
model:
  type: gguf|openai|vllm
  path: /path/to/model.gguf  # For GGUF
  endpoint: http://localhost:8000  # For OpenAI/vLLM
  model_name: model-name  # For OpenAI/vLLM
  n_gpu_layers: -1  # For GGUF
  n_ctx: 8192  # Context window
  max_tokens: 512
  temperature: 0.7
  is_vlm: false  # Vision Language Model

focus:
  type: internal|conversational|functional
  id: ajson://senter/focuses/<focus_name>
  name: <Focus Display Name>
  created: YYYY-MM-DDTHH:MM:SSZ
  version: 1.0

system_prompt: |
  [Multi-line system prompt defining the agent's purpose, behavior, and capabilities]
  
  ## Your Mission
  [What this agent does]
  
  ## Your Expertise
  [Specific capabilities]
  
  ## Capabilities
  [What the agent can do]
  
  ## Response Style
  [How the agent should respond]
  
  ## Output Format
  [Expected output format, e.g., JSON for internal agents]

---

# Context Sections (Optional - parsed by agents)

## User Preferences
[To be populated by Profiler agent]

## Patterns Observed
[To be populated by Profiler agent]

## Goals & Objectives
[To be populated by Goal_Detector agent]

## Evolution Notes
[To be populated by Profiler agent]

## Function Metadata (for Functional Focuses)
functions:
  - name: function_name
    description: What it does
    parameters: [list of parameters]
    returns: What it returns

## Tool Information (for Tool Focuses)
tool_name: <name>
tool_path: /path/to/tool/script
usage_examples:
  - Example 1
  - Example 2
