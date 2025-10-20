"""
This conftest file contains global fixtures and parameters to be used
in the following test.
"""

from copy import deepcopy

import pytest

# Building path to export and import files


@pytest.fixture(scope="function")
def tmp_function_file(tmp_path_factory):
    return tmp_path_factory.mktemp("function-test")


@pytest.fixture(scope="class")
def tmp_class_file(tmp_path_factory):
    return tmp_path_factory.mktemp("class-test")


_SYSTEM_CONFIG = {
    ("env", "production"): {
        "database": {
            "host": "prod-db.company.com",
            "port": 5432,
            "pools": [5, 10, 15],
            "replicas": {
                1: {"region": "us-east", "status": "active", "id": 42},
                2: {"region": "eu-west", "status": "standby", "id": 54},
            },
            # Configuration instances (CI) pour production
            "instances": {
                42: {
                    "name": "prod-primary",
                    "max_connections": 1000,
                    "type": "primary",
                    "maintenance_window": "02:00-04:00 UTC",
                },
                54: {
                    "name": "prod-secondary",
                    "max_connections": 800,
                    "type": "read_replica",
                    "sync_lag": "< 1s",
                },
            },
        },
        "api": {"rate_limit": 10000, "timeout": 30},
    },
    ("env", "dev"): {
        "database": {
            "host": "dev-db.internal.com",
            "port": 5433,
            "pools": [2, 5, 8],
            "replicas": {
                1: {"region": "us-east", "status": "active", "id": 12},
                2: {"region": "eu-west", "status": "standby", "id": 34},
            },
            "backup_frequency": "daily",
            "instances": {
                12: {
                    "name": "dev-main",
                    "max_connections": 200,
                    "type": "development",
                    "auto_cleanup": True,
                    "reset_schedule": "weekly",
                },
                34: {
                    "name": "dev-testing",
                    "max_connections": 150,
                    "type": "testing",
                    "isolation_level": "READ_UNCOMMITTED",
                    "ephemeral": True,
                },
            },
        },
        "api": {"rate_limit": 1000, "timeout": 60, "debug_mode": True},
        "features": {
            "experimental": ["new_auth", "beta_ui"],
            "flags": {"enable_logging": True, "mock_external_apis": True},
        },
    },
    frozenset(["cache", "redis"]): {
        "nodes": ["cache-1", "cache-2"],
        "config": {"ttl": 3600, "memory": "2GB"},
        "environments": {
            ("env", "production"): {
                "cluster_size": 6,
                "persistence": "rdb",
                "max_memory_policy": "allkeys-lru",
            },
            ("env", "dev"): {
                "cluster_size": 2,
                "persistence": "none",
                "max_memory_policy": "volatile-lru",
            },
        },
    },
    "monitoring": {
        ("metrics", "cpu"): [80, 90, 95],  # seuils d'alerte
        ("logs", "level"): {"error": "/var/log/error.log", "debug": None},
        "dashboards": {
            ("env", "production"): {
                "grafana_url": "https://monitoring.company.com",
                "alerts": ["slack", "pagerduty"],
                "retention": "1 year",
            },
            ("env", "dev"): {
                "grafana_url": "http://dev-monitoring.internal.com",
                "alerts": ["email"],
                "retention": "30 days",
            },
        },
    },
    # Configuration globale des environnements
    "global_settings": {
        ("security", "encryption"): {
            "algorithm": "AES-256-GCM",
            "key_rotation": {("env", "production"): 90, ("env", "dev"): 365},  # jours
        },
        "security": {"encryption": "mandatory", "level": 100},
        "networking": {
            "load_balancer": {
                ("env", "production"): {
                    "type": "AWS ALB",
                    "instances": 3,
                    "health_check_interval": 30,
                },
                ("env", "dev"): {
                    "type": "nginx",
                    "instances": 1,
                    "health_check_interval": 60,
                },
            }
        },
    },
}


@pytest.fixture(scope="function")
def function_system_config():
    return deepcopy(_SYSTEM_CONFIG)


@pytest.fixture(scope="class")
def class_system_config():
    # Provide a fresh deep copy for each test class to avoid shared-state mutations
    return deepcopy(_SYSTEM_CONFIG)
