"""Node-specific planning diagnostics and capacity-based queue recommendations."""

from dataclasses import replace

import pytest

from falcon.models import NodeResources
from falcon.planning import plan_resources
from falcon.quantities import parse_cpu, parse_memory_gib


def node(name="gpu-node", **kwargs):
    return replace(NodeResources(name, 64, 4, 256, 32, 4, 3, "2080ti"), **kwargs)


def plan(nodes, **kwargs):
    return plan_resources(nodes, "2080ti", "2080ti", 1, **kwargs)


def test_gpu_only_request_reports_current_headroom_and_node_share_baseline():
    result = plan([node()])
    info = result.scheduling
    assert info["gpu_available_nodes"] == ["gpu-node"]
    assert info["schedulable_nodes"] == ["gpu-node"]
    assert info["maximum_allocation"]["basis"] == "currently_available"
    assert info["maximum_allocation"]["cpu_cores"] == 60
    assert info["maximum_allocation"]["memory_gib"] == 224
    assert info["recommended_allocation"] == {
        "node": "gpu-node", "cpu": "16", "memory": "63Gi", "basis": "currently_available",
    }


def test_busy_gpus_recommend_from_capacity_not_current_free_cpu_or_memory():
    result = plan([node(gpu_used=4, cpu_used=63, memory_used_gib=255)])
    info = result.scheduling
    assert not result.immediately_schedulable
    assert not info["gpu_available_now"]
    assert info["maximum_allocation"]["basis"] == "node_capacity"
    assert info["maximum_allocation"]["cpu_cores"] == 64
    assert info["maximum_allocation"]["memory_gib"] == 256
    assert info["recommended_allocation"]["cpu"] == "16"
    assert info["recommended_allocation"]["memory"] == "63Gi"
    assert info["recommended_allocation"]["basis"] == "node_capacity"
    assert info["next_action"] == "wait_for_capacity"
    assert "submit unchanged" in info["guidance"]
    assert info["nodes"][0]["shortfall_now"] == {"gpus": 1, "cpu_cores": 15, "memory_gib": 62}


def test_free_gpus_keep_recommendations_capped_to_current_headroom():
    info = plan([node(cpu_used=62, memory_used_gib=252)]).scheduling
    assert info["recommended_allocation"]["basis"] == "currently_available"
    assert info["recommended_allocation"]["cpu"] == "2"
    assert info["recommended_allocation"]["memory"] == "3Gi"


@pytest.mark.parametrize("cpu,memory,blocked,shortfall", [
    ("61", "8Gi", "cpu_cores", 1), ("1", "225Gi", "memory_gib", 1),
])
def test_warning_identifies_exact_resource_shortfall(cpu, memory, blocked, shortfall):
    result = plan([node()], cpu_override=cpu, memory_override=memory)
    row = result.scheduling["nodes"][0]
    assert row["blocking_resources"] == [blocked]
    assert row["shortfall_now"][blocked] == shortfall
    assert result.scheduling["fits_any_node_capacity"]
    assert result.scheduling["next_action"] == "choose_alternative_or_wait"
    assert "short by 1" in result.warning


def test_impossible_cpu_memory_request_is_distinguished_from_waitable_shortage():
    result = plan([node()], cpu_override="80", memory_override="512Gi")
    assert not result.scheduling["fits_any_node_capacity"]
    assert result.scheduling["nodes"][0]["shortfall_capacity"] == {
        "gpus": 0, "cpu_cores": 16, "memory_gib": 256,
    }
    assert result.scheduling["next_action"] == "revise_request"
    assert "waiting cannot make it fit" in result.warning
    assert "waiting alone will not fix it" in result.scheduling["guidance"]


def test_maxima_from_different_nodes_are_not_a_jointly_feasible_option():
    nodes = [node("cpu-rich", cpu_total=32, cpu_used=0, memory_total_gib=8, memory_used_gib=0),
             node("ram-rich", cpu_total=2, cpu_used=0, memory_total_gib=64, memory_used_gib=0)]
    info = plan(nodes, cpu_override="4", memory_override="16Gi").scheduling
    maximum = info["maximum_allocation"]
    assert maximum["cpu_cores"] == 32
    assert maximum["memory_gib"] == 64
    assert maximum["cpu_nodes"] == ["cpu-rich"]
    assert maximum["memory_nodes"] == ["ram-rich"]
    assert not info["fits_any_node_capacity"]
    assert not info["schedulable_nodes"]
    assert "different nodes" in maximum["note"]
    for option in info["alternatives"]:
        target = next(item for item in nodes if item.name == option["node"])
        assert float(parse_cpu(option["cpu"])) <= target.cpu_free
        assert parse_memory_gib(option["memory"]) <= target.memory_free_gib
        assert option["gpu"] == {"model": "2080ti", "count": 1}


def test_no_zero_or_oversized_alternatives_when_cpu_or_memory_is_exhausted():
    for target in [node(cpu_used=64), node(memory_used_gib=256)]:
        info = plan([target], cpu_override="1", memory_override="1Gi").scheduling
        assert info["alternatives"] == []
        assert not info["schedulable_now"]


def test_explicit_shared_memory_is_not_silently_broken_by_smaller_options():
    info = plan([node(memory_used_gib=248)], cpu_override="1", memory_override="32Gi", shared_memory="20Gi").scheduling
    assert info["alternatives"] == []


def test_unknown_inventory_does_not_claim_free_resources():
    result = plan([node(scheduling_info_available=False)])
    assert not result.immediately_schedulable
    assert result.scheduling["gpu_available_now"] is None
    assert result.scheduling["nodes"][0]["available"] is None
    assert result.scheduling["next_action"] == "recheck_inventory"


def test_available_gpu_nodes_not_busy_gpu_nodes_supply_current_maxima():
    info = plan([node("available"), node("busy", cpu_total=256, gpu_used=4)]).scheduling
    assert info["maximum_allocation"]["cpu_cores"] == 60
    assert info["gpu_available_nodes"] == ["available"]
    busy = next(row for row in info["nodes"] if row["node"] == "busy")
    assert busy["recommended_allocation"]["basis"] == "node_capacity"


def test_gpu_limits_are_per_node_not_cluster_totals():
    info = plan([node("a", gpu_total=4, gpu_used=1), node("b", gpu_total=8, gpu_used=7)]).scheduling
    assert info["gpu_limits"]["max_per_node"] == 8
    assert info["gpu_limits"]["max_free_per_node"] == 3
    assert info["gpu_limits"]["single_node_required"]
