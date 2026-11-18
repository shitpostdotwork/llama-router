"""
Checks how a model's "gpus" field becomes one GPU mask per replica

A nested list places each replica on its own GPUs while residency still counts the union
The test spawns no process and needs no GPU, so it runs in the default `pytest` collection
"""

import json
from pathlib import Path

from router import LLMRouter


def make_router(tmp_path: Path, llm: dict) -> LLMRouter:
    """
    A router over a three GPU host whose LLM section is llm

    Args:
        tmp_path (Path) : The pytest scratch directory the config files go in
        llm (dict)      : The config's LLM section

    Returns:
        The router, never started
    """
    config = {
        "LLM": llm,
        "ROUTER": {"NUM_GPUS": 3},
        "LLM-base-port": 30000,
        "llama-server-executable": "/bin/false",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    presets_path = tmp_path / "presets.ini"
    presets_path.write_text("[*]\n")
    return LLMRouter(str(config_path), str(presets_path))


def test_nested_gpus_place_one_replica_per_mask(tmp_path: Path):
    r = make_router(tmp_path, {"swarm": {"num_instance": 3, "gpus": [[0], [1], [2]]}})
    assert r.model_gpus["swarm"] == [0, 1, 2]
    assert r.model_placements["swarm"] == [[0], [1], [2]]
    assert r._replica_masks("swarm", 3) == [[0], [1], [2]]


def test_flat_gpus_share_the_union(tmp_path: Path):
    r = make_router(tmp_path, {
        "split": {"num_instance": 2, "gpus": [0, 2]},
        "unpinned": {"num_instance": 1},
        "everywhere": {"num_instance": 1, "gpus": "all"},
    })
    assert r.model_placements["split"] == [[0, 2]]
    assert r._replica_masks("split", 2) == [[0, 2], [0, 2]]
    assert r.model_placements["unpinned"] == [[0]]
    assert r.model_placements["everywhere"] == [[0, 1, 2]]


def test_masks_cycle_and_mix_bare_ids(tmp_path: Path):
    r = make_router(tmp_path, {"swarm": {"num_instance": 4, "gpus": [[0, 1], 2]}})
    assert r.model_gpus["swarm"] == [0, 1, 2]
    assert r._replica_masks("swarm", 4) == [[0, 1], [2], [0, 1], [2]]


def test_top_up_respawns_only_the_lost_mask(tmp_path: Path):
    r = make_router(tmp_path, {"swarm": {"num_instance": 3, "gpus": [[0], [1], [2]]}})
    # Fake two live replicas on GPUs 0 and 2; the one on GPU 1 died
    for port, gpus in ((30000, [0]), (30002, [2])):
        r.processes[port] = object()  # type: ignore[assignment]
        r.port_model[port] = "swarm"
        r.port_gpus[port] = gpus
    assert r._replica_masks("swarm", 3) == [[1]]


def test_reserved_ports_are_not_handed_out_twice(tmp_path: Path):
    r = make_router(tmp_path, {"m": {"num_instance": 1}})
    first = r._alloc_ports(2)
    second = r._alloc_ports(1)
    assert first == [30000, 30001]
    assert second == [30002]
    r._unreserve_ports(first)
    assert r._alloc_ports(1) == [30000]
