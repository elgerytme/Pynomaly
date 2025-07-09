#!/usr/bin/env python3
"""
Start Redis server for development and testing.
This script handles Redis installation and startup for different platforms.
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import urllib.request
import zipfile
import shutil

def is_redis_installed():
    """Check if Redis is installed and accessible."""
    try:
        result = subprocess.run(['redis-server', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_redis_windows():
    """Install Redis on Windows."""
    print("Installing Redis for Windows...")
    
    # Download Redis for Windows
    redis_url = "https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip"
    redis_dir = Path.home() / "redis"
    redis_zip = redis_dir / "redis.zip"
    
    # Create redis directory
    redis_dir.mkdir(exist_ok=True)
    
    # Download Redis
    print(f"Downloading Redis from {redis_url}...")
    urllib.request.urlretrieve(redis_url, redis_zip)
    
    # Extract Redis
    print("Extracting Redis...")
    with zipfile.ZipFile(redis_zip, 'r') as zip_ref:
        zip_ref.extractall(redis_dir)
    
    # Clean up
    redis_zip.unlink()
    
    # Add to PATH (for current session)
    os.environ['PATH'] = str(redis_dir) + os.pathsep + os.environ['PATH']
    
    print(f"Redis installed in {redis_dir}")
    return redis_dir / "redis-server.exe"

def install_redis_linux():
    """Install Redis on Linux."""
    print("Installing Redis on Linux...")
    
    # Try different package managers
    package_managers = [
        ['apt-get', 'update', '&&', 'apt-get', 'install', '-y', 'redis-server'],
        ['yum', 'install', '-y', 'redis'],
        ['dnf', 'install', '-y', 'redis'],
        ['pacman', '-S', '--noconfirm', 'redis']
    ]
    
    for cmd in package_managers:
        try:
            subprocess.run(cmd, check=True)
            print("Redis installed successfully")
            return 'redis-server'
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("Failed to install Redis using package manager")
    return None

def install_redis_macos():
    """Install Redis on macOS."""
    print("Installing Redis on macOS...")
    
    # Try brew first
    try:
        subprocess.run(['brew', 'install', 'redis'], check=True)
        print("Redis installed successfully with Homebrew")
        return 'redis-server'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Homebrew not found or failed")
        return None

def start_redis_server(redis_executable=None):
    """Start Redis server."""
    if redis_executable is None:
        redis_executable = 'redis-server'
    
    print(f"Starting Redis server with {redis_executable}...")
    
    # Redis configuration
    redis_config = {
        'port': '6379',
        'bind': '127.0.0.1',
        'protected-mode': 'no',
        'save': '900 1',
        'save': '300 10',
        'save': '60 10000',
        'rdbcompression': 'yes',
        'dbfilename': 'dump.rdb',
        'dir': str(Path.cwd()),
        'maxmemory': '256mb',
        'maxmemory-policy': 'allkeys-lru',
        'appendonly': 'yes',
        'appendfsync': 'everysec'
    }
    
    # Create temporary config file
    config_file = Path.cwd() / 'redis.conf'
    with open(config_file, 'w') as f:
        for key, value in redis_config.items():
            f.write(f"{key} {value}\n")
    
    try:
        # Start Redis server
        process = subprocess.Popen([
            str(redis_executable),
            str(config_file)
        ])
        
        # Wait a bit for server to start
        time.sleep(2)
        
        # Test connection
        try:
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'PONG' in result.stdout:
                print("✅ Redis server started successfully!")
                print(f"📊 Redis is running on port 6379")
                print(f"🔧 Config file: {config_file}")
                print(f"📁 Data directory: {Path.cwd()}")
                print("\nTo stop Redis, press Ctrl+C or kill the process")
                return process
            else:
                print("❌ Redis server failed to start properly")
                return None
        except subprocess.TimeoutExpired:
            print("❌ Redis server is not responding")
            return None
            
    except FileNotFoundError:
        print(f"❌ Redis executable not found: {redis_executable}")
        return None
    except Exception as e:
        print(f"❌ Error starting Redis: {e}")
        return None

def main():
    """Main function to start Redis server."""
    print("🚀 Starting Redis Server for Pynomaly Development")
    print("=" * 50)
    
    # Check if Redis is already installed
    if not is_redis_installed():
        print("Redis not found. Installing...")
        
        system = platform.system().lower()
        redis_executable = None
        
        if system == 'windows':
            redis_executable = install_redis_windows()
        elif system == 'linux':
            redis_executable = install_redis_linux()
        elif system == 'darwin':  # macOS
            redis_executable = install_redis_macos()
        else:
            print(f"❌ Unsupported operating system: {system}")
            sys.exit(1)
        
        if redis_executable is None:
            print("❌ Failed to install Redis")
            sys.exit(1)
    else:
        print("✅ Redis is already installed")
        redis_executable = 'redis-server'
    
    # Start Redis server
    process = start_redis_server(redis_executable)
    
    if process is None:
        print("❌ Failed to start Redis server")
        sys.exit(1)
    
    try:
        # Keep the process running
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Redis server...")
        process.terminate()
        process.wait()
        
        # Clean up config file
        config_file = Path.cwd() / 'redis.conf'
        if config_file.exists():
            config_file.unlink()
            
        print("✅ Redis server stopped successfully")

if __name__ == '__main__':
    main()
