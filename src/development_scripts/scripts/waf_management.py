#!/usr/bin/env python3
"""
Web Application Firewall (WAF) Management Script
Provides comprehensive WAF administration capabilities.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import redis
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly_detection.infrastructure.config import Settings
from src.anomaly_detection.infrastructure.security.waf_middleware import (
    AttackType,
    ThreatLevel,
    ThreatSignature,
    WAFMiddleware,
)

console = Console()
logger = logging.getLogger(__name__)


class WAFManager:
    """WAF management interface."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client = redis.from_url(settings.redis_url)
        self.waf_middleware = WAFMiddleware(None, settings)
        self.config_path = Path("config/security/waf_config.json")
        self.signatures_path = Path("config/security/waf_signatures.json")

    def get_status(self) -> dict[str, Any]:
        """Get WAF status and statistics."""
        return self.waf_middleware.get_stats()

    def list_blocked_ips(self) -> list[dict[str, Any]]:
        """List all blocked IPs."""
        blocked_ips = []

        # Get from Redis
        try:
            keys = self.redis_client.keys("waf:blocked:*")
            for key in keys:
                ip = key.decode("utf-8").split(":")[-1]
                data = self.redis_client.get(key)
                if data:
                    block_info = json.loads(data)
                    blocked_ips.append(
                        {
                            "ip": ip,
                            "reason": block_info.get("reason", "Unknown"),
                            "timestamp": block_info.get("timestamp", 0),
                            "duration": block_info.get("duration", 0),
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to get blocked IPs: {e}")

        return blocked_ips

    def list_suspicious_ips(self) -> list[dict[str, Any]]:
        """List IPs with reputation scores."""
        suspicious_ips = []

        try:
            keys = self.redis_client.keys("waf:reputation:*")
            for key in keys:
                ip = key.decode("utf-8").split(":")[-1]
                score = self.redis_client.get(key)
                if score:
                    suspicious_ips.append(
                        {
                            "ip": ip,
                            "reputation_score": int(score),
                            "risk_level": self._get_risk_level(int(score)),
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to get suspicious IPs: {e}")

        return suspicious_ips

    def _get_risk_level(self, score: int) -> str:
        """Get risk level based on reputation score."""
        if score >= 80:
            return "Critical"
        elif score >= 60:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"

    async def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        return await self.waf_middleware.unblock_ip(ip)

    async def block_ip(
        self, ip: str, reason: str = "Manual block", duration: int = 3600
    ) -> bool:
        """Block an IP address."""
        try:
            await self.waf_middleware._block_ip(ip, reason)
            console.print(f"✅ IP {ip} blocked successfully", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Failed to block IP {ip}: {e}", style="red")
            return False

    def list_signatures(self) -> list[ThreatSignature]:
        """List all threat signatures."""
        return self.waf_middleware.signatures

    def add_signature(self, signature: ThreatSignature) -> bool:
        """Add a new threat signature."""
        try:
            # Load existing signatures
            if self.signatures_path.exists():
                with open(self.signatures_path) as f:
                    data = json.load(f)
            else:
                data = {"signatures": []}

            # Add new signature
            sig_data = {
                "name": signature.name,
                "pattern": signature.pattern,
                "attack_type": signature.attack_type.value,
                "threat_level": signature.threat_level.value,
                "description": signature.description,
                "enabled": signature.enabled,
                "custom": True,
            }

            data["signatures"].append(sig_data)

            # Save updated signatures
            with open(self.signatures_path, "w") as f:
                json.dump(data, f, indent=2)

            console.print(
                f"✅ Signature '{signature.name}' added successfully", style="green"
            )
            return True

        except Exception as e:
            console.print(f"❌ Failed to add signature: {e}", style="red")
            return False

    def update_signature(self, name: str, enabled: bool) -> bool:
        """Update signature status."""
        try:
            if self.signatures_path.exists():
                with open(self.signatures_path) as f:
                    data = json.load(f)

                for sig in data["signatures"]:
                    if sig["name"] == name:
                        sig["enabled"] = enabled
                        break
                else:
                    console.print(f"❌ Signature '{name}' not found", style="red")
                    return False

                with open(self.signatures_path, "w") as f:
                    json.dump(data, f, indent=2)

                console.print(
                    f"✅ Signature '{name}' updated successfully", style="green"
                )
                return True

        except Exception as e:
            console.print(f"❌ Failed to update signature: {e}", style="red")
            return False

    def update_config(self, config_updates: dict[str, Any]) -> bool:
        """Update WAF configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            config.update(config_updates)

            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            console.print("✅ Configuration updated successfully", style="green")
            return True

        except Exception as e:
            console.print(f"❌ Failed to update configuration: {e}", style="red")
            return False

    def get_recent_attacks(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent attack attempts."""
        attacks = []

        try:
            # Get from WAF middleware stats
            stats = self.waf_middleware.stats
            for attack in list(stats.recent_attacks)[-limit:]:
                attacks.append(
                    {
                        "timestamp": datetime.fromtimestamp(
                            attack.timestamp
                        ).isoformat(),
                        "ip": attack.ip,
                        "attack_type": attack.attack_type.value,
                        "threat_level": attack.threat_level.value,
                        "signature": attack.signature,
                        "blocked": attack.blocked,
                        "risk_score": attack.risk_score,
                        "details": attack.details,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to get recent attacks: {e}")

        return attacks

    def test_signature(self, pattern: str, test_input: str) -> bool:
        """Test a signature pattern against input."""
        try:
            import re

            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            return bool(compiled_pattern.search(test_input))
        except Exception as e:
            console.print(f"❌ Pattern test failed: {e}", style="red")
            return False

    def generate_report(self, hours: int = 24) -> dict[str, Any]:
        """Generate WAF security report."""
        stats = self.get_status()
        blocked_ips = self.list_blocked_ips()
        suspicious_ips = self.list_suspicious_ips()
        recent_attacks = self.get_recent_attacks()

        # Calculate time-based metrics
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_attacks_in_window = [
            attack
            for attack in recent_attacks
            if datetime.fromisoformat(attack["timestamp"]) > cutoff_time
        ]

        return {
            "report_period": f"{hours} hours",
            "generated_at": datetime.now().isoformat(),
            "overall_stats": stats,
            "blocked_ips_count": len(blocked_ips),
            "suspicious_ips_count": len(suspicious_ips),
            "recent_attacks_count": len(recent_attacks_in_window),
            "top_attack_types": self._get_top_attack_types(recent_attacks_in_window),
            "top_attackers": self._get_top_attackers(recent_attacks_in_window),
            "threat_level_distribution": self._get_threat_distribution(
                recent_attacks_in_window
            ),
            "blocked_ips": blocked_ips,
            "suspicious_ips": suspicious_ips,
            "recent_attacks": recent_attacks_in_window,
        }

    def _get_top_attack_types(self, attacks: list[dict[str, Any]]) -> dict[str, int]:
        """Get top attack types from recent attacks."""
        attack_types = {}
        for attack in attacks:
            attack_type = attack["attack_type"]
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1

        return dict(sorted(attack_types.items(), key=lambda x: x[1], reverse=True))

    def _get_top_attackers(self, attacks: list[dict[str, Any]]) -> dict[str, int]:
        """Get top attacking IPs from recent attacks."""
        attackers = {}
        for attack in attacks:
            ip = attack["ip"]
            attackers[ip] = attackers.get(ip, 0) + 1

        return dict(sorted(attackers.items(), key=lambda x: x[1], reverse=True)[:10])

    def _get_threat_distribution(self, attacks: list[dict[str, Any]]) -> dict[str, int]:
        """Get threat level distribution from recent attacks."""
        levels = {}
        for attack in attacks:
            level = attack["threat_level"]
            levels[level] = levels.get(level, 0) + 1

        return levels


def print_status_table(status: dict[str, Any]):
    """Print WAF status table."""
    table = Table(title="WAF Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    waf_stats = status["waf_stats"]
    table.add_row("Uptime", f"{waf_stats['uptime_seconds']:.0f} seconds")
    table.add_row("Total Requests", str(waf_stats["total_requests"]))
    table.add_row("Blocked Requests", str(waf_stats["blocked_requests"]))
    table.add_row("Attacks Detected", str(waf_stats["attacks_detected"]))
    table.add_row("Requests/Second", f"{waf_stats['requests_per_second']:.2f}")
    table.add_row("Attack Rate", f"{waf_stats['attack_rate']:.2f}%")
    table.add_row("Blocked IPs", str(status["blocked_ips"]))
    table.add_row("Suspicious IPs", str(status["suspicious_ips"]))
    table.add_row("Active Signatures", str(status["active_signatures"]))

    console.print(table)


def print_ip_table(ips: list[dict[str, Any]], title: str):
    """Print IP table."""
    if not ips:
        console.print(f"No {title.lower()} found", style="yellow")
        return

    table = Table(title=title)
    table.add_column("IP Address", style="cyan")
    table.add_column("Score/Reason", style="magenta")
    table.add_column("Status", style="green")

    for ip_info in ips:
        if "reputation_score" in ip_info:
            score_reason = f"Score: {ip_info['reputation_score']}"
            status = ip_info["risk_level"]
        else:
            score_reason = ip_info["reason"]
            status = "Blocked"

        table.add_row(ip_info["ip"], score_reason, status)

    console.print(table)


def print_signatures_table(signatures: list[ThreatSignature]):
    """Print signatures table."""
    table = Table(title="Threat Signatures")
    table.add_column("Name", style="cyan")
    table.add_column("Attack Type", style="magenta")
    table.add_column("Threat Level", style="red")
    table.add_column("Enabled", style="green")
    table.add_column("Custom", style="yellow")

    for sig in signatures:
        enabled = "✅" if sig.enabled else "❌"
        custom = "✅" if sig.custom else "❌"
        table.add_row(
            sig.name, sig.attack_type.value, sig.threat_level.value, enabled, custom
        )

    console.print(table)


def print_attacks_table(attacks: list[dict[str, Any]]):
    """Print recent attacks table."""
    if not attacks:
        console.print("No recent attacks found", style="yellow")
        return

    table = Table(title="Recent Attacks")
    table.add_column("Time", style="cyan")
    table.add_column("IP", style="magenta")
    table.add_column("Attack Type", style="red")
    table.add_column("Threat Level", style="yellow")
    table.add_column("Blocked", style="green")
    table.add_column("Risk Score", style="blue")

    for attack in attacks[-20:]:  # Show last 20 attacks
        timestamp = datetime.fromisoformat(attack["timestamp"])
        blocked = "✅" if attack["blocked"] else "❌"
        table.add_row(
            timestamp.strftime("%H:%M:%S"),
            attack["ip"],
            attack["attack_type"],
            attack["threat_level"],
            blocked,
            str(attack["risk_score"]),
        )

    console.print(table)


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="WAF Management Tool")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show WAF status")

    # IP management commands
    ip_parser = subparsers.add_parser("ip", help="IP management")
    ip_subparsers = ip_parser.add_subparsers(dest="ip_command")

    list_blocked_parser = ip_subparsers.add_parser(
        "list-blocked", help="List blocked IPs"
    )
    list_suspicious_parser = ip_subparsers.add_parser(
        "list-suspicious", help="List suspicious IPs"
    )

    block_parser = ip_subparsers.add_parser("block", help="Block IP")
    block_parser.add_argument("ip", help="IP address to block")
    block_parser.add_argument("--reason", default="Manual block", help="Block reason")
    block_parser.add_argument(
        "--duration", type=int, default=3600, help="Block duration in seconds"
    )

    unblock_parser = ip_subparsers.add_parser("unblock", help="Unblock IP")
    unblock_parser.add_argument("ip", help="IP address to unblock")

    # Signature management commands
    sig_parser = subparsers.add_parser("signatures", help="Signature management")
    sig_subparsers = sig_parser.add_subparsers(dest="sig_command")

    list_sig_parser = sig_subparsers.add_parser("list", help="List signatures")

    add_sig_parser = sig_subparsers.add_parser("add", help="Add signature")
    add_sig_parser.add_argument("name", help="Signature name")
    add_sig_parser.add_argument("pattern", help="Regex pattern")
    add_sig_parser.add_argument(
        "attack_type", choices=[t.value for t in AttackType], help="Attack type"
    )
    add_sig_parser.add_argument(
        "threat_level", choices=[t.value for t in ThreatLevel], help="Threat level"
    )
    add_sig_parser.add_argument("--description", default="", help="Description")

    enable_sig_parser = sig_subparsers.add_parser("enable", help="Enable signature")
    enable_sig_parser.add_argument("name", help="Signature name")

    disable_sig_parser = sig_subparsers.add_parser("disable", help="Disable signature")
    disable_sig_parser.add_argument("name", help="Signature name")

    test_sig_parser = sig_subparsers.add_parser("test", help="Test signature")
    test_sig_parser.add_argument("pattern", help="Regex pattern")
    test_sig_parser.add_argument("input", help="Test input")

    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    set_config_parser = config_subparsers.add_parser("set", help="Set configuration")
    set_config_parser.add_argument("key", help="Configuration key")
    set_config_parser.add_argument("value", help="Configuration value")

    # Attack monitoring commands
    attacks_parser = subparsers.add_parser("attacks", help="Attack monitoring")
    attacks_subparsers = attacks_parser.add_subparsers(dest="attacks_command")

    list_attacks_parser = attacks_subparsers.add_parser(
        "list", help="List recent attacks"
    )
    list_attacks_parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of attacks"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate security report")
    report_parser.add_argument(
        "--hours", type=int, default=24, help="Report time window in hours"
    )
    report_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.verbose:
        # Use secure logging configuration instead of direct debug logging
        try:
            from src.packages.data.anomaly_detection.core.security_configuration import get_security_config, configure_secure_logging
            security_config = get_security_config()
            if security_config.is_development():
                configure_secure_logging()
            else:
                logging.basicConfig(level=logging.INFO)
        except ImportError:
            # Fallback if security config not available
            logging.basicConfig(level=logging.INFO)

    # Initialize WAF manager
    settings = Settings()
    waf_manager = WAFManager(settings)

    if args.command == "status":
        status = waf_manager.get_status()
        print_status_table(status)

    elif args.command == "ip":
        if args.ip_command == "list-blocked":
            blocked_ips = waf_manager.list_blocked_ips()
            print_ip_table(blocked_ips, "Blocked IPs")

        elif args.ip_command == "list-suspicious":
            suspicious_ips = waf_manager.list_suspicious_ips()
            print_ip_table(suspicious_ips, "Suspicious IPs")

        elif args.ip_command == "block":
            await waf_manager.block_ip(args.ip, args.reason, args.duration)

        elif args.ip_command == "unblock":
            success = await waf_manager.unblock_ip(args.ip)
            if success:
                console.print(f"✅ IP {args.ip} unblocked successfully", style="green")
            else:
                console.print(f"❌ Failed to unblock IP {args.ip}", style="red")

    elif args.command == "signatures":
        if args.sig_command == "list":
            signatures = waf_manager.list_signatures()
            print_signatures_table(signatures)

        elif args.sig_command == "add":
            signature = ThreatSignature(
                name=args.name,
                pattern=args.pattern,
                attack_type=AttackType(args.attack_type),
                threat_level=ThreatLevel(args.threat_level),
                description=args.description,
                custom=True,
            )
            waf_manager.add_signature(signature)

        elif args.sig_command == "enable":
            waf_manager.update_signature(args.name, True)

        elif args.sig_command == "disable":
            waf_manager.update_signature(args.name, False)

        elif args.sig_command == "test":
            result = waf_manager.test_signature(args.pattern, args.input)
            if result:
                console.print("✅ Pattern matches input", style="green")
            else:
                console.print("❌ Pattern does not match input", style="red")

    elif args.command == "config":
        if args.config_command == "set":
            # Parse value as JSON if possible
            try:
                value = json.loads(args.value)
            except:
                value = args.value

            waf_manager.update_config({args.key: value})

    elif args.command == "attacks":
        if args.attacks_command == "list":
            attacks = waf_manager.get_recent_attacks(args.limit)
            print_attacks_table(attacks)

    elif args.command == "report":
        report = waf_manager.generate_report(args.hours)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            console.print(f"✅ Report saved to {args.output}", style="green")
        else:
            console.print(Panel(JSON.from_data(report), title="WAF Security Report"))

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
