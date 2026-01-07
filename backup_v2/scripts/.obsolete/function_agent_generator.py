#!/usr/bin/env python3
"""
Function Agent Generator
Automatically creates JSON agent manifests for Python scripts in the Functions/ directory
Checks for existing agents to avoid duplicates
"""

import os
import sys
import json
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from senter_selector import SenterSelector

class FunctionAgentGenerator:
    def __init__(self, functions_dir: str = "Functions", agents_dir: str = "Agents", topics_dir: str = "Topics"):
        self.functions_dir = Path(functions_dir)
        self.agents_dir = Path(agents_dir)
        self.topics_dir = Path(topics_dir)
        self.agents_dir.mkdir(exist_ok=True)
        self.topics_dir.mkdir(exist_ok=True)
        self.selector = SenterSelector()

    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file to extract function information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or ""
                    })

            module_docstring = ast.get_docstring(tree) or ""

            return {
                'functions': functions,
                'docstring': module_docstring,
                'filename': file_path.name,
                'filepath': str(file_path)
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {
                'functions': [],
                'docstring': f"Python script: {file_path.name}",
                'filename': file_path.name,
                'filepath': str(file_path)
            }

    def get_existing_agents(self) -> List[Dict[str, Any]]:
        """Get list of existing agent manifests"""
        agents = []
        if self.agents_dir.exists():
            for json_file in self.agents_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        agent_data = json.load(f)
                        agent_data['_filename'] = json_file.name
                        agents.append(agent_data)
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
        return agents

    def check_agent_exists(self, script_name: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Check if an agent already exists for this functionality
        Returns the existing agent filename if found, None otherwise
        """
        existing_agents = self.get_existing_agents()

        if not existing_agents:
            return None

        # Create a description of what this script does
        script_description = analysis['docstring'] or f"Functionality for {script_name}"

        # Get list of existing agent descriptions
        agent_options = []
        for agent in existing_agents:
            agent_name = agent['agent']['name']
            agent_desc = agent['agent']['description']
            agent_tags = agent['agent'].get('tags', [])
            agent_options.append(f"{agent_name}: {agent_desc} (tags: {', '.join(agent_tags)})")

        # Use the selector to find if there's already a matching agent
        try:
            selected_option, reasoning = self.selector.select_from_options(
                query=f"Find an existing agent that provides: {script_description}",
                options=agent_options,
                max_final_options=3,
                allow_new=True,
                context=f"Script: {script_name}, Description: {script_description}"
            )

            if selected_option and selected_option != "CREATE NEW":
                # Extract the agent name from the selected option
                agent_name = selected_option.split(':')[0].strip()
                # Find the corresponding filename
                for agent in existing_agents:
                    if agent['agent']['name'] == agent_name:
                        return agent['_filename']

        except Exception as e:
            print(f"Error checking for existing agent: {e}")

        return None

    def generate_agent_manifest(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a JSON agent manifest from the analysis"""
        filename = analysis['filename']
        script_name = filename.replace('.py', '')

        agent_id = f"ajson://ai-toolbox/functions/{script_name}"

        capabilities = []
        tools = []

        for func in analysis['functions']:
            cap_id = func['name'].lower().replace('_', '-')
            capabilities.append({
                "id": cap_id,
                "description": func['docstring'][:100] if func['docstring'] else f"Execute {func['name']} function"
            })

            tool_params = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for arg in func['args']:
                if arg != 'self':
                    tool_params["properties"][arg] = {
                        "type": "string",
                        "description": f"Parameter {arg}"
                    }
                    tool_params["required"].append(arg)

            tools.append({
                "id": func['name'],
                "type": "function",
                "description": func['docstring'] or f"Execute {func['name']} function",
                "function": {
                    "name": func['name'],
                    "description": func['docstring'] or f"Execute {func['name']} function",
                    "parameters": tool_params
                }
            })

        if not capabilities:
            capabilities.append({
                "id": "execute-script",
                "description": analysis['docstring'][:100] or f"Execute {script_name} script"
            })

        manifest = {
            "manifest_version": "1.0",
            "profiles": ["core", "exec"],
            "agent": {
                "id": agent_id,
                "name": script_name.replace('_', ' ').title(),
                "version": "1.0.0",
                "description": analysis['docstring'] or f"Agent for {script_name} functionality",
                "author": "AI Toolbox",
                "tags": ["function", "script", script_name]
            },
            "capabilities": capabilities,
            "tools": tools,
            "modalities": {
                "input": ["text"],
                "output": ["text"]
            },
            "runtime": {
                "type": "python",
                "version": ">=3.8",
                "entrypoint": analysis['filename']
            }
        }

        return manifest

    def create_topic_for_agent(self, script_name: str, analysis: Dict[str, Any], agent_filename: str):
        """Create a topic directory and SENTER.md for the new agent"""
        topic_dir = self.topics_dir / script_name
        topic_dir.mkdir(exist_ok=True)

        # Create SENTER.md with initial context
        senter_md = topic_dir / "SENTER.md"
        if not senter_md.exists():
            content = f"""# {script_name.replace('_', ' ').title()}

## Overview
This topic covers the {script_name} functionality, providing access to specialized functions and capabilities.

## Agent
- **Agent**: {agent_filename.replace('.json', '')}
- **Description**: {analysis['docstring'] or f"Agent for {script_name} functionality"}

## Functions
"""
            for func in analysis['functions']:
                content += f"- **{func['name']}**: {func['docstring'] or 'Execute function'}\n"

            content += """
## Usage Patterns
- Add usage examples and context here as the system learns

## Context Updates
- System will automatically update this file with conversation context
"""

            with open(senter_md, 'w', encoding='utf-8') as f:
                f.write(content)

        # Update topic_agent_map.json
        map_file = Path("config/topic_agent_map.json")
        topic_agent_map = {}
        if map_file.exists():
            try:
                with open(map_file, 'r') as f:
                    topic_agent_map = json.load(f)
            except:
                pass

        # Add mapping
        topic_agent_map[script_name] = agent_filename.replace('.json', '')

        with open(map_file, 'w') as f:
            json.dump(topic_agent_map, f, indent=2)

    def generate_agents_for_functions(self):
        """Scan Functions directory and generate agents for all Python scripts"""
        if not self.functions_dir.exists():
            print(f"Functions directory {self.functions_dir} does not exist")
            return

        python_files = list(self.functions_dir.glob("*.py"))
        print(f"Found {len(python_files)} Python files in {self.functions_dir}")

        generated_count = 0
        skipped_count = 0

        for py_file in python_files:
            print(f"Analyzing {py_file.name}...")

            # Analyze the file
            analysis = self.analyze_python_file(py_file)

            # Check if agent already exists
            script_name = py_file.stem
            existing_agent = self.check_agent_exists(script_name, analysis)

            if existing_agent:
                print(f"Skipping {py_file.name} - agent already exists: {existing_agent}")
                skipped_count += 1
                continue

            # Generate manifest
            manifest = self.generate_agent_manifest(analysis)

            # Save to Agents directory
            agent_filename = f"{script_name}.json"
            agent_path = self.agents_dir / agent_filename

            with open(agent_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # Create corresponding topic
            self.create_topic_for_agent(script_name, analysis, agent_filename)

            print(f"Generated agent: {agent_path}")
            generated_count += 1

        print(f"Generated {generated_count} new agents, skipped {skipped_count} existing ones")

def main():
    generator = FunctionAgentGenerator()
    generator.generate_agents_for_functions()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">ai-toolbox/Senter/scripts/function_agent_generator.py