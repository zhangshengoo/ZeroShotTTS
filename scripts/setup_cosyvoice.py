#!/usr/bin/env python3
"""
CosyVoice安装和启动脚本
"""
import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    """运行shell命令"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Output: {result.stdout}")
    return True

def install_cosyvoice():
    """安装CosyVoice"""
    print("🚀 开始安装CosyVoice...")
    
    # 检查路径
    cosyvoice_path = "TTS_Model/CosyVoice"
    if not os.path.exists(cosyvoice_path):
        print("❌ CosyVoice子模块未找到，请先运行: git submodule update --init --recursive")
        return False
    
    # 安装依赖
    print("📦 安装CosyVoice依赖...")
    requirements_file = os.path.join(cosyvoice_path, "requirements.txt")
    if os.path.exists(requirements_file):
        if not run_command(f"{sys.executable} -m pip install -r {requirements_file}"):
            return False
    else:
        print("⚠️  requirements.txt未找到，使用基础依赖")
        base_deps = [
            "torch",
            "torchaudio",
            "transformers",
            "librosa",
            "numpy",
            "scipy",
            "soundfile",
            "fastapi",
            "uvicorn"
        ]
        if not run_command(f"{sys.executable} -m pip install {' '.join(base_deps)}"):
            return False
    
    print("✅ CosyVoice安装完成！")
    return True

def start_cosyvoice_service():
    """启动CosyVoice服务"""
    print("🔧 启动CosyVoice服务...")
    
    cosyvoice_path = "TTS_Model/CosyVoice"
    
    # 检查API服务脚本
    api_script = os.path.join(cosyvoice_path, "cosyvoice", "cli", "cosyvoice_server.py")
    if os.path.exists(api_script):
        cmd = f"{sys.executable} {api_script} --port 50000"
    else:
        # 使用通用启动方式
        cmd = f"{sys.executable} -m fastapi dev --host 0.0.0.0 --port 50000"
    
    print(f"启动命令: {cmd}")
    print("服务将运行在: http://localhost:50000")
    print("API文档: http://localhost:50000/docs")
    
    try:
        subprocess.run(cmd, shell=True, cwd=cosyvoice_path)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_cosyvoice()
    else:
        start_cosyvoice_service()

if __name__ == "__main__":
    main()