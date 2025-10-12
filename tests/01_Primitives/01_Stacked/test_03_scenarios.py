import re

import pytest

from ndict_tools import StackedKeyError
from ndict_tools.tools import _StackedDict


class TestScenario01StrictSD:

    @pytest.mark.parametrize(
        "pop_item_result",
        [
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                60,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                1,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "nginx",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                30,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                3,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                ["global_settings", "networking", "load_balancer"],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
            (
                ["global_settings", "networking"],
                _StackedDict(default_setup={"indent": 2, "default_factory": None}),
            ),
        ],
    )
    def test_pop_item(self, strict_c_sd, pop_item_result):
        assert strict_c_sd.popitem() == pop_item_result

    @pytest.mark.parametrize(
        "keys, pop_result",
        [
            (
                ["global_settings", "security"],
                _StackedDict(
                    {"encryption": "mandatory", "level": 100},
                    default_setup={"indent": 2, "default_factory": None},
                ),
            ),
            (
                "global_settings",
                _StackedDict(
                    {
                        ("security", "encryption"): {
                            "algorithm": "AES-256-GCM",
                            "key_rotation": {
                                ("env", "production"): 90,
                                ("env", "dev"): 365,
                            },
                        }
                    },
                    default_setup={"indent": 2, "default_factory": None},
                ),
            ),
        ],
    )
    def test_pop(self, strict_c_sd, keys, pop_result):
        assert strict_c_sd.pop(keys) == pop_result

    @pytest.mark.parametrize(
        "keys, error_smg",
        [
            (
                ["global_settings", "security"],
                "Key path ['global_settings', 'security'] does not exist. (key: ['global_settings', 'security'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'])",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'])",
            ),
        ],
    )
    def test_pop_failed(self, strict_c_sd, keys, error_smg):
        with pytest.raises(StackedKeyError, match=re.escape(error_smg)):
            strict_c_sd.pop(keys)

    def test_update(self, strict_c_sd):
        strict_c_sd.update(
            {
                "global_settings": {
                    ("security", "encryption"): {
                        "algorithm": "AES-256-GCM",
                        "key_rotation": {
                            ("env", "production"): 90,
                            ("env", "dev"): 365,
                        },  # jours
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
                }
            }
        )
        assert strict_c_sd[["global_settings", "security", "encryption"]] == "mandatory"


class TestScenario01SmoothSD:

    @pytest.mark.parametrize(
        "pop_item_result",
        [
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                60,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                1,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "nginx",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                30,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                3,
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "AWS ALB",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                ["global_settings", "networking", "load_balancer"],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
            (
                ["global_settings", "networking"],
                _StackedDict(
                    default_setup={"indent": 2, "default_factory": _StackedDict}
                ),
            ),
        ],
    )
    def test_pop_item(self, smooth_c_sd, pop_item_result):
        assert smooth_c_sd.popitem() == pop_item_result

    @pytest.mark.parametrize(
        "keys, pop_result",
        [
            (
                ["global_settings", "security"],
                _StackedDict(
                    {"encryption": "mandatory", "level": 100},
                    default_setup={"indent": 2, "default_factory": _StackedDict},
                ),
            ),
            (
                "global_settings",
                _StackedDict(
                    {
                        ("security", "encryption"): {
                            "algorithm": "AES-256-GCM",
                            "key_rotation": {
                                ("env", "production"): 90,
                                ("env", "dev"): 365,
                            },
                        }
                    },
                    default_setup={"indent": 2, "default_factory": _StackedDict},
                ),
            ),
        ],
    )
    def test_pop(self, smooth_c_sd, keys, pop_result):
        assert smooth_c_sd.pop(keys) == pop_result

    @pytest.mark.parametrize(
        "keys, error_smg",
        [
            (
                ["global_settings", "security"],
                "Key path ['global_settings', 'security'] does not exist. (key: ['global_settings', 'security'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'type'])",
            ),
            (
                ["global_settings", "networking", "load_balancer", ("env", "dev")],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'health_check_interval'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "instances",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'instances'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                    "type",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production'), 'type'])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "production"),
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'production')] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'production')])",
            ),
            (
                [
                    "global_settings",
                    "networking",
                    "load_balancer",
                    ("env", "dev"),
                    "health_check_interval",
                ],
                "Key path ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'] does not exist. (key: ['global_settings', 'networking', 'load_balancer', ('env', 'dev'), 'health_check_interval'])",
            ),
        ],
    )
    def test_pop_failed(self, smooth_c_sd, keys, error_smg):
        with pytest.raises(StackedKeyError, match=re.escape(error_smg)):
            smooth_c_sd.pop(keys)

    def test_update(self, smooth_c_sd):
        smooth_c_sd.update(
            {
                "global_settings": {
                    ("security", "encryption"): {
                        "algorithm": "AES-256-GCM",
                        "key_rotation": {
                            ("env", "production"): 90,
                            ("env", "dev"): 365,
                        },  # jours
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
                }
            }
        )
        assert smooth_c_sd[["global_settings", "security", "encryption"]] == "mandatory"
