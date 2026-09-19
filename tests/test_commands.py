"""Pure /swarm argument-tail contracts; no runtime, storage, or live install.

Run with the integrated host interpreter used by the other plugin tests.
"""
from itertools import permutations
import argparse
import sqlite3
import subprocess

import pytest

import swarm


@pytest.fixture(autouse=True)
def parse_only(tmp_path, monkeypatch):
    """Fail even if a parser catches an attempted runtime/storage operation."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("parse_command must not perform runtime/storage operations")

    for attribute in ("store_for", "allowed_store", "launch_member", "SwarmStore", "RecordStore"):
        monkeypatch.setattr(swarm, attribute, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    # ArgumentParser may exit or normalize a payload; this grammar must not use it.
    monkeypatch.setattr(argparse.ArgumentParser, "__init__", forbidden)
    for key in (
        "AGENT_ZOO_HOME", "AGENT_ZOO_STATE_ROOT", "AGENT_ZOO_LOCAL_STATE_ROOT",
        "AZO_TUI_STORE_ROOT", "AZO_TUI_LIVE_SESSION_STORE_ROOT", "AGENT_ZOO_INSTALL_ROOT",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    ):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    for key in ("AZO_SWARM_ID", "AZO_SWARM_POD_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    yield
    assert not calls, "parsing attempted a forbidden operation"
    assert not list(tmp_path.iterdir()), "parsing wrote filesystem artifacts"


def test_create_example():
    assert swarm.parse_command("-n 16 -p 4 --channels comms,breakthroughs --name explore") == {
        "action": "create", "agents": 16, "pods": 4,
        "channels": ("comms", "breakthroughs"), "name": "explore",
    }


@pytest.mark.parametrize("raw", ["-n 1", "  -n 1  ", "\n\t-n\t1\r\n", "-n1"])
def test_create_defaults_and_surrounding_whitespace(raw):
    assert swarm.parse_command(raw) == {
        "action": "create", "agents": 1, "pods": 1,
        "channels": ("general",), "name": None,
    }


_CREATE_ORDERS = list(permutations(("-n 16", "-p 4", "--channels comms,breakthroughs", "--name explore")))


@pytest.mark.parametrize("parts", _CREATE_ORDERS, ids=[f"order-{i}" for i in range(len(_CREATE_ORDERS))])
def test_create_flags_in_any_order(parts):
    assert swarm.parse_command(" ".join(parts)) == {
        "action": "create", "agents": 16, "pods": 4,
        "channels": ("comms", "breakthroughs"), "name": "explore",
    }


@pytest.mark.parametrize("raw", [
    "-n16 -p4 --channels=comms,breakthroughs --name=explore",
    "--name=explore -p4 --channels comms,breakthroughs -n 16",
    "--channels=comms,breakthroughs -n16 --name explore -p 4",
])
def test_create_attached_and_equals_values(raw):
    assert swarm.parse_command(raw) == {
        "action": "create", "agents": 16, "pods": 4,
        "channels": ("comms", "breakthroughs"), "name": "explore",
    }


@pytest.mark.parametrize("agents,pods", [(64, 1), (32, 32), (2048, 32)])
def test_create_inclusive_topology_limits(agents, pods):
    assert swarm.parse_command(f"-n{agents} -p{pods}") == {
        "action": "create", "agents": agents, "pods": pods,
        "channels": ("general",), "name": None,
    }


@pytest.mark.parametrize("identifier", ["A0_b-c", "a" * 64], ids=["mixed", "max-length"])
def test_create_valid_identifier_boundaries(identifier):
    result = swarm.parse_command(f"-n1 --name={identifier} --channels={identifier},general")
    assert result["name"] == identifier
    assert result["channels"] == (identifier, "general")


@pytest.mark.parametrize("raw", ["", " \n\t", "-p 1", "--name explore", "--channels general"])
def test_create_requires_agents(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", [
    "-n0", "-n-1", "-n 1 -p0", "-n 1 -p -1", "-n 1.5", "-n one",
    "-n 1 -p 1.5", "-n 1 -p one", "-n 1e2",
])
def test_create_rejects_nonpositive_or_noninteger_counts(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", ["-n 3 -p 2", "-n 1 -p 2", "-n 33 -p 33", "-n 65", "-n 130 -p 2"])
def test_create_rejects_uneven_or_excessive_topology(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", ["-n", "-n 1 -p", "-n 1 --channels", "-n 1 --name", "-n -p 1"])
def test_create_rejects_missing_option_values(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", ["-n1 --bogus", "-n1 -x2", "-n1 --help", "-n1 stray", "-n1 --chan general"])
def test_create_rejects_unknown_flags_and_extra_tokens(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", [
    "-n1 -n 1", "-n1 -n2", "-n2 -p1 -p 1",
    "-n1 --channels=general --channels general", "-n1 --name=x --name x",
])
def test_create_rejects_duplicate_flags_even_when_values_match(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("channels", ["", ",", ",a", "a,", "a,,b", "a,a", "a,b,a", "../bad", "a,b/c"])
def test_create_rejects_empty_duplicate_or_invalid_channels(channels):
    with pytest.raises(ValueError):
        swarm.parse_command(f"-n1 --channels={channels}")


@pytest.mark.parametrize("identifier", ["", "../bad", "a/b", "a.b", "-bad", "_bad", "a" * 65],
                         ids=["empty", "parent-path", "slash", "dot", "leading-hyphen", "leading-underscore", "too-long"])
def test_create_rejects_invalid_names(identifier):
    with pytest.raises(ValueError):
        swarm.parse_command(f"-n1 --name={identifier}")


@pytest.mark.parametrize("raw", ["create -n1", "create", "status", "bogus sw0", "/swarm -n1"])
def test_rejects_unknown_verbs_explicit_create_and_non_tail_input(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw,target", [
    ('bcast "message"', None), ("  bcast 'message' \n", None),
    ('bcast sw0 "message"', "sw0"),
])
def test_bcast_default_selection_and_optional_target(raw, target):
    assert swarm.parse_command(raw) == {
        "action": "bcast", "target": target, "pods": None, "message": "message",
    }


@pytest.mark.parametrize("prefix,target", [
    ("bcast sw0 -p 0-3,5", "sw0"), ("bcast -p 0-3,5 sw0", "sw0"),
    ("bcast -p 0-3,5", None),
])
def test_bcast_target_before_or_after_pod_selection(prefix, target):
    assert swarm.parse_command(f'{prefix} "message"') == {
        "action": "bcast", "target": target, "pods": (0, 1, 2, 3, 5), "message": "message",
    }


@pytest.mark.parametrize("selector,expected", [
    ("5,0-3,2,5,3-4", (0, 1, 2, 3, 4, 5)), ("0-0", (0,)),
    ("31", (31,)), ("31,0", (0, 31)), ("0-31", tuple(range(32))),
], ids=["sorted-union", "singleton-range", "max-index", "sparse", "all-pods"])
def test_bcast_selector_union_is_sorted_deduplicated_and_topology_independent(selector, expected):
    assert swarm.parse_command(f'bcast missing-swarm -p {selector} "message"') == {
        "action": "bcast", "target": "missing-swarm", "pods": expected, "message": "message",
    }


@pytest.mark.parametrize("selector", [
    "-1", "0--1", "3-0", "1:3", "0-3:2", "1.0", "1e1", "x", "0-1-2",
    "-", "1-", ",1", "1,", "1,,2", "32", "0-32", "999999999999999999999",
])
def test_bcast_rejects_invalid_selectors_and_indices_above_31(selector):
    with pytest.raises(ValueError):
        swarm.parse_command(f'bcast -p {selector} "message"')


@pytest.mark.parametrize("raw", [
    'bcast -p "message"', 'bcast -p 0 -p 1 "message"',
    'bcast sw0 sw1 "message"', 'bcast --bogus "message"',
    'bcast --name sw0 "message"',
])
def test_bcast_rejects_missing_duplicate_or_unknown_options_and_extra_targets(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("quote,encoded,expected", [
    ('"', r'say \"hello\"', 'say "hello"'),
    ("'", r"it\'s fine", "it's fine"),
    ('"', r'C:\\tmp\\file', r'C:\tmp\file'),
    ("'", r'C:\\tmp\\file', r'C:\tmp\file'),
    ('"', r'literal \n \t \u263a \q', r'literal \n \t \u263a \q'),
    ("'", r'literal \n \t \u263a \q', r'literal \n \t \u263a \q'),
    ('"', r"opposite \' stays", r"opposite \' stays"),
    ("'", r'opposite \" stays', r'opposite \" stays'),
    ('"', r'ends in \\', 'ends in \\'),
], ids=["double-quote", "single-quote", "double-backslashes", "single-backslashes",
        "double-literals", "single-literals", "double-opposite", "single-opposite", "final-backslash"])
def test_bcast_decodes_only_matching_quote_and_backslash(quote, encoded, expected):
    assert swarm.parse_command(f"bcast {quote}{encoded}{quote}")["message"] == expected


@pytest.mark.parametrize("quote,opposite", [('"', "'"), ("'", '"')], ids=["double", "single"])
def test_bcast_preserves_whitespace_real_newlines_unicode_and_opposite_quotes(quote, opposite):
    payload = f"  first\n\t雪 🐝 café {opposite}quoted{opposite}\r\nlast  "
    assert swarm.parse_command(f"\n bcast {quote}{payload}{quote}\t \n") == {
        "action": "bcast", "target": None, "pods": None, "message": payload,
    }


def test_bcast_payload_flags_are_not_parsed_as_options():
    payload = "-p 32 --name wrong --channels=, --bogus cancel sw0"
    assert swarm.parse_command(f'bcast sw0 -p 0 "{payload}"') == {
        "action": "bcast", "target": "sw0", "pods": (0,), "message": payload,
    }


def test_bcast_supports_multiline_payload_over_64_kib():
    payload = (" x \n" * 30000) + "終"
    assert len(payload) > 120000
    assert swarm.parse_command(f'bcast "{payload}"') == {
        "action": "bcast", "target": None, "pods": None, "message": payload,
    }


@pytest.mark.parametrize("raw", ["bcast", "bcast sw0", "bcast -p 0", "bcast sw0 hello", "bcast hello world"])
def test_bcast_requires_one_quoted_message(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", ['bcast ""', "bcast ''", 'bcast " \n\t\r "', "bcast '  '"])
def test_bcast_rejects_empty_and_whitespace_only_messages(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("raw", [
    'bcast "unterminated', "bcast 'unterminated", 'bcast "mismatch\'',
    'bcast "trailing escape' + "\\", 'bcast "escaped closing' + '\\"',
    'bcast "ok" trailing', 'bcast "ok" -p 0', 'bcast "ok" "again"',
    'bcast "ok"suffix', 'bcast "ok"\'again\'',
], ids=["double-open", "single-open", "mismatched", "dangling-escape", "escaped-closer",
        "trailing-word", "trailing-option", "two-payloads", "suffix", "adjacent-payloads"])
def test_bcast_rejects_unterminated_quotes_and_tokens_after_payload(raw):
    with pytest.raises(ValueError):
        swarm.parse_command(raw)


@pytest.mark.parametrize("verb", ["cancel", "interrupt", "continue"])
def test_control_verbs_return_only_action_and_target(verb):
    assert swarm.parse_command(f" \n{verb}\t sw0 \n") == {"action": verb, "target": "sw0"}


@pytest.mark.parametrize("verb", ["cancel", "interrupt", "continue", "release", "capture"])
@pytest.mark.parametrize("tail", ["sw0 sw1", "sw0 -p 0", "--bogus", "../bad"],
                         ids=["two-targets", "extra-option", "unknown-option", "invalid-id"])
def test_control_verbs_reject_extra_or_invalid_targets(verb, tail):
    with pytest.raises(ValueError):
        swarm.parse_command(f"{verb} {tail}")


@pytest.mark.parametrize("verb", ["interrupt", "continue"])
def test_lifecycle_resume_pause_still_require_explicit_target(verb):
    with pytest.raises(ValueError):
        swarm.parse_command(verb)
