#!/usr/bin/env python3
"""
Buck2 Remote Caching Setup Script
Configures Buck2 remote caching for the Monorepo monorepo
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

def update_buckconfig_cache(
    cache_type: str,
    address: str,
    token: Optional[str] = None,
    additional_config: Optional[Dict[str, str]] = None
) -> None:
    """Update .buckconfig with remote cache configuration"""
    
    buckconfig_path = Path(".buckconfig")
    if not buckconfig_path.exists():
        print("❌ .buckconfig file not found in current directory")
        sys.exit(1)
    
    # Read existing config
    with open(buckconfig_path, 'r') as f:
        lines = f.readlines()
    
    # Find cache section and update
    new_lines = []
    in_cache_section = False
    cache_section_found = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('[cache]'):
            in_cache_section = True
            cache_section_found = True
            new_lines.append(line)
            continue
        
        if in_cache_section and stripped.startswith('['):
            # End of cache section
            if cache_type == "http":
                new_lines.append(f"\n[cache.http]\n")
                new_lines.append(f"  address = {address}\n")
                if token:
                    new_lines.append(f"  read_headers = Authorization: Bearer {token}\n")
                    new_lines.append(f"  write_headers = Authorization: Bearer {token}\n")
                if additional_config:
                    for key, value in additional_config.items():
                        new_lines.append(f"  {key} = {value}\n")
                new_lines.append("\n")
            elif cache_type == "re":
                new_lines.append(f"\n[cache.re]\n")
                new_lines.append(f"  action_cache_address = {address}\n")
                if additional_config:
                    for key, value in additional_config.items():
                        new_lines.append(f"  {key} = {value}\n")
                new_lines.append("\n")
            
            in_cache_section = False
        
        if not (in_cache_section and stripped.startswith(f'[cache.{cache_type}]')):
            new_lines.append(line)
    
    if not cache_section_found:
        print("❌ [cache] section not found in .buckconfig")
        sys.exit(1)
    
    # Write updated config
    with open(buckconfig_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated .buckconfig with {cache_type} cache configuration")

def setup_github_actions_cache() -> None:
    """Setup GitHub Actions cache configuration"""
    
    # Create environment-specific cache config
    cache_config = """# GitHub Actions Buck2 Cache Configuration
# Add this to your GitHub Actions workflow

env:
  BUCK2_CACHE_REPO: ${{ github.repository }}
  BUCK2_CACHE_BRANCH: ${{ github.ref_name }}
  BUCK2_CACHE_KEY: buck2-${{ runner.os }}-${{ hashFiles('BUCK', '**/*.bzl', 'third-party/**/*') }}

steps:
  - name: Cache Buck2 outputs
    uses: actions/cache@v4
    with:
      path: |
        .buck-out
        .buck-cache
      key: ${{ env.BUCK2_CACHE_KEY }}
      restore-keys: |
        buck2-${{ runner.os }}-
        
  - name: Build with Buck2 cache
    run: |
      buck2 build //... \\
        --config cache.mode=dir \\
        --config cache.dir=.buck-cache
"""
    
    cache_config_path = Path(".github/cache/buck2-cache-config.yml")
    cache_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_config_path, 'w') as f:
        f.write(cache_config)
    
    print(f"✅ Created GitHub Actions cache configuration at {cache_config_path}")

def setup_docker_cache() -> None:
    """Setup Docker-based remote cache"""
    
    docker_compose = """# Docker-based Buck2 Remote Cache
version: '3.8'

services:
  buck2-cache:
    image: redis:7-alpine
    container_name: buck2-cache
    ports:
      - "6379:6379"
    volumes:
      - buck2_cache_data:/data
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    
  buck2-cache-web:
    image: nginx:alpine
    container_name: buck2-cache-web
    ports:
      - "8080:80"
    volumes:
      - ./nginx-cache.conf:/etc/nginx/nginx.conf
    depends_on:
      - buck2-cache

volumes:
  buck2_cache_data:
"""
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream cache_backend {
        server buck2-cache:6379;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location /cache/ {
            proxy_pass http://cache_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        location /health {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }
}
"""
    
    docker_path = Path("docker/buck2-cache/docker-compose.yml")
    docker_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(docker_path, 'w') as f:
        f.write(docker_compose)
    
    with open(docker_path.parent / "nginx-cache.conf", 'w') as f:
        f.write(nginx_config)
    
    print(f"✅ Created Docker cache setup at {docker_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Setup Buck2 remote caching for Monorepo monorepo"
    )
    parser.add_argument(
        "--type", 
        choices=["http", "re", "github", "docker"],
        required=True,
        help="Type of remote cache to setup"
    )
    parser.add_argument(
        "--address",
        help="Cache server address (for http/re types)"
    )
    parser.add_argument(
        "--token",
        help="Authentication token (for http type)"
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Additional config in key=value format"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Setting up Buck2 {args.type} remote cache...")
    
    if args.type in ["http", "re"]:
        if not args.address:
            print(f"❌ --address is required for {args.type} cache type")
            sys.exit(1)
        
        additional_config = {}
        if args.config:
            for config_item in args.config:
                if "=" not in config_item:
                    print(f"❌ Invalid config format: {config_item}. Use key=value")
                    sys.exit(1)
                key, value = config_item.split("=", 1)
                additional_config[key] = value
        
        update_buckconfig_cache(args.type, args.address, args.token, additional_config)
    
    elif args.type == "github":
        setup_github_actions_cache()
    
    elif args.type == "docker":
        setup_docker_cache()
    
    print(f"✅ Buck2 {args.type} remote cache setup complete!")
    print("\n📋 Next steps:")
    
    if args.type == "http":
        print("1. Configure your HTTP cache server endpoint")
        print("2. Set BUCK2_CACHE_TOKEN environment variable")
        print("3. Test with: buck2 build //... --config cache.http.address=YOUR_ENDPOINT")
    
    elif args.type == "re":
        print("1. Configure your Remote Execution endpoints")
        print("2. Set up authentication as needed")
        print("3. Test with: buck2 build //... --config cache.re.action_cache_address=YOUR_ENDPOINT")
    
    elif args.type == "github":
        print("1. Add the cache configuration to your GitHub Actions workflows")
        print("2. Update BUCK2_CACHE_KEY environment variable as needed")
        print("3. Monitor cache hit rates in Actions logs")
    
    elif args.type == "docker":
        print("1. Run: docker-compose -f docker/buck2-cache/docker-compose.yml up -d")
        print("2. Update .buckconfig with: address = http://localhost:8080/cache/")
        print("3. Test with: buck2 build //...")

if __name__ == "__main__":
    main()