#!/usr/bin/env python3
"""
Senter CLI Build Script - Compiles senter_cli.cpp using llama.cpp build system
Simplified version that actually works with llama.cpp API
"""

import subprocess
import sys
from pathlib import Path


def main():
    senter_root = Path(__file__).parent.parent
    llama_cpp = senter_root.parent / "ai-toolbox" / "Resources" / "llama.cpp"

    # Check if llama.cpp exists
    if not llama_cpp.exists():
        print("❌ llama.cpp not found at:", llama_cpp)
        print("   Expected: /home/sovthpaw/ai-toolbox/Resources/llama.cpp")
        sys.exit(1)

    print("🔨 Building Senter Native CLI...")
    print(f"   llama.cpp: {llama_cpp}")
    print(f"   Senter root: {senter_root}")

    # Build directory
    build_dir = senter_root / "build_senter_cli"
    build_dir.mkdir(exist_ok=True)

    # Simple CMake configuration
    print("\n📝 Configuring CMake...")
    cmake_cmd = [
        "cmake",
        f"-S{llama_cpp}",
        f"-B{build_dir}",
        "-DLLAMA_CLI=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    result = subprocess.run(cmake_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("❌ CMake configuration failed!")
        sys.exit(1)

    print("✅ CMake configuration complete")

    # Build
    print("\n🔨 Compiling senter_cli...")
    build_cmd = ["cmake", "--build", build_dir]
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        print("❌ Build failed!")
        sys.exit(1)

    print("✅ Build complete!")

    # Copy binary to Senter/scripts/
    bin_name = "senter_cli"
    if sys.platform == "win32":
        bin_name += ".exe"

    src_bin = build_dir / "bin" / bin_name
    dst_bin = senter_root / "scripts" / bin_name

    if not src_bin.exists():
        print(f"❌ Binary not found: {src_bin}")
        sys.exit(1)

    print(f"\n📦 Copying {bin_name} to scripts/...")
    subprocess.run(["cp", str(src_bin), str(dst_bin)])

    # Make executable
    subprocess.run(["chmod", "+x", str(dst_bin)])

    print("✅ Senter CLI binary ready!")
    print(f"\n🚀 Run with: {dst_bin}")
    print("\n💡 Note: This is a simple version that compiles with llama.cpp")
    print("💡 For full functionality, use: python3 scripts/senter.py")


if __name__ == "__main__":
    main()
