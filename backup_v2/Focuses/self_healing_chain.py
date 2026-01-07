#!/usr/bin/env python3
"""
Self-Healing Chain - Error Detection and Fix System
Detects errors from function outputs and triggers Planner → Coder chain
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from senter_md_parser import SenterMdParser
from omniagent import SenterOmniAgent


class SelfHealingChain:
    """
    Chain for self-healing: Error Detection → Planner → Coder → Update Focus

    Detects errors from function outputs and automatically fixes them
    """

    def __init__(self, senter_root: Path):
        self.senter_root = senter_root
        self.parser = SenterMdParser(senter_root)

        # Load internal agents for self-healing
        self.planner = self._load_internal_agent("Planner_Agent")
        self.coder = self._load_internal_agent("Coder_Agent")
        self.diagnostic = self._load_internal_agent("Diagnostic_Agent")

    def _load_internal_agent(self, agent_name: str) -> Optional[SenterOmniAgent]:
        """Load internal agent as omniagent instance"""
        try:
            internal_dir = self.senter_root / "Focuses" / "internal" / agent_name
            if not internal_dir.exists():
                return None

            # Load internal agent config
            config = self.parser.load_focus_config(f"internal/{agent_name}")

            # Internal agents use same omniagent infrastructure
            omni = SenterOmniAgent(
                model_config=config.get("model", {}),
                omni_config=self._get_infrastructure_config(),
                embed_config=self._get_embed_config(),
            )

            return omni
        except Exception as e:
            print(f"   ⚠️  Failed to load internal agent {agent_name}: {e}")
            return None

    def _get_infrastructure_config(self) -> dict:
        """Get infrastructure models config (Omni 3B + embedding)"""
        config_file = self.senter_root / "config" / "senter_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("infrastructure_models", {})
        return {}

    def _get_embed_config(self) -> dict:
        """Get embedding model config"""
        config_file = self.senter_root / "config" / "senter_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("infrastructure_models", {}).get(
                    "embedding_model", {}
                )
        return {}

    def process_function_output(
        self, focus_name: str, output: str, error: bool = False
    ):
        """
        Process function output, detect errors, trigger fix chain if needed

        Args:
            focus_name: Focus with function
            output: Function output
            error: Was there an error?

        Returns:
            Action taken (none, fixing, fixed)
        """
        if not error:
            return "none"  # No error, no action needed

        print(f"   🔧 Error detected in Focus: {focus_name}")
        print(f"   📄 Error output: {output[:200]}...")

        # Stage 1: Self-inference with problem
        problem_analysis = self._analyze_problem(focus_name, output)

        if not problem_analysis:
            return "none"

        # Stage 2: Planner creates fix plan
        fix_plan = self._create_fix_plan(focus_name, problem_analysis)

        if not fix_plan:
            return "none"

        # Stage 3: Coder writes fix
        fix_code = self._write_fix_code(focus_name, fix_plan)

        if not fix_code:
            return "none"

        # Stage 4: Update Focus context with fix
        self._update_focus_with_fix(focus_name, fix_code)

        return "fixed"

    def _analyze_problem(
        self, focus_name: str, error_output: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use Diagnostic_Agent to analyze the problem
        """
        if not self.diagnostic:
            print(f"   ⚠️  Diagnostic_Agent not available, skipping analysis")
            return None

        diagnostic_prompt = f"""
        Analyze this error from Focus function:

        Focus: {focus_name}
        Error Output:
        {error_output}

        Your task:
        1. Identify the type of error (syntax, runtime, configuration, API, etc.)
        2. Determine the root cause
        3. Assess the severity and impact
        4. Recommend a course of action

        Respond with JSON:
        {{
          "error_type": "syntax|runtime|configuration|api|network|unknown",
          "root_cause": "description of root cause",
          "severity": "low|medium|high|critical",
          "course_of_action": "what should be done to fix"
        }}
        """

        try:
            response = self.diagnostic.process_text(diagnostic_prompt, max_tokens=512)
            return self._parse_diagnostic_response(response)
        except Exception as e:
            print(f"   ⚠️  Diagnostic analysis failed: {e}")
            return None

    def _parse_diagnostic_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse diagnostic response JSON"""
        try:
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"   ⚠️  Failed to parse diagnostic response: {e}")
            return None

    def _create_fix_plan(
        self, focus_name: str, problem_analysis: Dict[str, Any]
    ) -> Optional[str]:
        """
        Use Planner_Agent to create step-by-step fix plan
        """
        if not self.planner:
            print(f"   ⚠️  Planner_Agent not available, skipping planning")
            return None

        plan_prompt = f"""
        Create a plan to fix this problem:

        Focus: {focus_name}
        Error Type: {problem_analysis.get("error_type", "unknown")}
        Root Cause: {problem_analysis.get("root_cause", "N/A")}
        Severity: {problem_analysis.get("severity", "unknown")}

        Recommended Action: {problem_analysis.get("course_of_action", "N/A")}

        Your task:
        Break down the fix into clear, achievable steps.
        Each step should be specific and actionable.

        Respond with step-by-step plan in plain text.
        """

        try:
            response = self.planner.process_text(plan_prompt, max_tokens=512)
            print(f"   📋 Fix plan created: {len(response)} chars")
            return response
        except Exception as e:
            print(f"   ⚠️  Fix plan creation failed: {e}")
            return None

    def _write_fix_code(self, focus_name: str, fix_plan: str) -> Optional[str]:
        """
        Use Coder_Agent to write the fix code
        """
        if not self.coder:
            print(f"   ⚠️  Coder_Agent not available, skipping code generation")
            return None

        code_prompt = f"""
        Write code to implement this fix:

        Focus: {focus_name}

        Fix Plan:
        {fix_plan}

        Your task:
        Write the complete code to implement this fix.
        Include proper error handling and logging.
        Respond with code block (```language).
        """

        try:
            response = self.coder.process_text(code_prompt, max_tokens=1024)
            print(f"   💻 Fix code generated: {len(response)} chars")
            return response
        except Exception as e:
            print(f"   ⚠️  Fix code generation failed: {e}")
            return None

    def _update_focus_with_fix(self, focus_name: str, fix_code: str):
        """
        Update Focus context with the fix
        """
        # Add fix to context section
        fix_entry = f"""
## Fix Applied: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Error Type:** Function execution error

**Fix Code:**
```
{fix_code}
```

**Notes:**
- Automatically applied by Self-Healing Chain
- Review and test before committing to production
"""

        self.parser.update_markdown_section(focus_name, "Fix Applied", fix_entry)

        print(f"   ✅ Focus {focus_name} updated with fix")

    def check_function_health(self, focus_name: str) -> Dict[str, Any]:
        """
        Check health of a Focus function

        Returns: health status dict
        """
        config = self.parser.load_focus_config(focus_name)
        functions = config.get("functions", [])

        if not functions:
            return {
                "status": "no_functions",
                "message": "Focus has no functions defined",
            }

        # Check if function scripts exist
        missing_scripts = []
        for func in functions:
            script_path = self.senter_root / func.get("script", "")
            if not script_path.exists():
                missing_scripts.append(func.get("name", "unknown"))

        if missing_scripts:
            return {
                "status": "missing_scripts",
                "message": f"Missing function scripts: {', '.join(missing_scripts)}",
            }

        return {
            "status": "healthy",
            "message": f"All {len(functions)} function(s) available",
        }


def main():
    """Test Self-Healing Chain"""
    senter_root = Path(__file__).parent.parent
    healing_chain = SelfHealingChain(senter_root)

    print("🔄 Starting Self-Healing Chain test...")

    # Test error processing
    test_error = """
    Error: Connection timeout after 30 seconds
    Traceback:
      File "wifi_lights.py", line 45
      raise ConnectionError("Failed to connect to lights hub")
    """

    result = healing_chain.process_function_output(
        "Wifi_Lights", test_error, error=True
    )

    print(f"\n✅ Self-Healing Chain test complete: {result}")


if __name__ == "__main__":
    main()
