import hashlib
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from falcon.agent_skills import (
    LEGACY_SKILL_BUNDLE_VERSION,
    METADATA_FILE,
    SKILL_BUNDLE_VERSION,
    SKILL_MODES,
    SUPPORTED_AGENTS,
    agent_skill_path,
    detect_agents,
    install_skill,
    install_skills,
    installed_skill_mode,
    refresh_installed_skills,
    uninstall_skill,
    uninstall_skills,
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bundle_digest(hashes):
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class AgentSkillTests(unittest.TestCase):
    def test_packaged_skill_is_one_concise_mcp_first_workflow(self):
        root = resources.files("falcon.skills").joinpath("falcon")
        skill = root.joinpath("SKILL.md").read_text(encoding="utf-8")
        frontmatter = skill.split("---", 2)[1]

        self.assertIn("name: falcon", frontmatter)
        self.assertIn("description:", frontmatter)
        self.assertNotIn("allowed-tools:", frontmatter)
        self.assertLessEqual(len(skill.splitlines()), 45)
        for redundant in (
            "falcon setup", "CLI is fallback", "interactive shell", "conda activate",
            "backoffLimit", "restartPolicy", "Coder Jobs", "permissions change", "exact reference/UID",
        ):
            self.assertNotIn(redundant, skill)
        for tool in (
            "plan_job", "submit_job", "wait_for_events", "job_logs",
            "job_metrics", "restart_job", "cancel_job",
        ):
            self.assertIn(f"`{tool}`", skill)
        self.assertIn("H100/PRO6000 use 75%", skill)
        self.assertIn("A6000/2080Ti use 10%", skill)
        self.assertIn("yield_time_ms=604800000", skill)
        for removed in (
            "falcon h100x2",
            "falcon logs JOB",
            "falcon metrics JOB",
            "falcon kill JOB",
            "falcon jobs",
            "falcon get",
            "falcon resources",
            "falcon top",
            "falcon delete",
            "reference.md",
        ):
            self.assertNotIn(removed, skill)

    def test_packaged_legacy_skill_preserves_direct_cli_workflow(self):
        root = resources.files("falcon.skills").joinpath("falcon-legacy")
        skill = root.joinpath("SKILL.md").read_text(encoding="utf-8")

        self.assertEqual(SKILL_MODES, ("mcp", "legacy"))
        self.assertIn("falcon logs JOB", skill)
        self.assertIn("falcon metrics JOB", skill)
        self.assertNotIn("plan_job", skill)

    def test_skill_explains_job_name_rules_without_renaming_arguments(self):
        skill = resources.files("falcon.skills").joinpath("falcon/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Job `name`", skill)
        self.assertIn("1–63 lowercase letters/digits/hyphens", skill)
        self.assertIn("starting and ending alphanumeric", skill)
        self.assertIn("replace underscores in the name only, not model arguments or paths", skill)

    def test_skill_sizes_resources_from_task_and_node_aware_baseline(self):
        skill = resources.files("falcon.skills").joinpath("falcon/SKILL.md").read_text(encoding="utf-8")
        for guidance in (
            "Request only CPU and host memory the task needs",
            "start `plan_job` with GPU model/count only",
            "node-aware recommended allocation as baseline",
            "less when sufficient",
            "Do not hard-code per-GPU CPU/RAM values",
            "scheduling.gpu_limits.max_per_node",
            "max_free_per_node",
            "never sum across nodes",
            "maxima/recommendations use node capacity",
            "Only exceed auto-allocation when absolutely necessary",
            "measured CPU bottlenecks",
            "GPU VRAM OOM is not evidence for more host RAM",
            "1 CPU and 1Gi suit trivial checks/smoke tests",
            "maximize GPU compute and VRAM utilization whenever the workload allows",
        ):
            self.assertIn(guidance, skill)

    def test_skill_keeps_host_specific_monitoring_without_a_wake_bridge(self):
        skill = resources.files("falcon.skills").joinpath("falcon/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("background report-only subagent", skill)
        self.assertIn("Claude Code 2.1.212+", skill)
        self.assertIn("call `wait_for_events` directly in the main conversation", skill)
        self.assertIn("let Claude Code background the long call", skill)
        self.assertNotIn("`haiku`", skill)
        self.assertIn("configured subagent model", skill)
        self.assertIn("async `wait_for_events` in `functions.exec`", skill)
        self.assertIn("cursor=0", skill)
        self.assertIn("monitoring as a separate explicit call", skill)
        self.assertIn("start one replacement async wait", skill)
        self.assertIn("same surviving cell", skill)
        self.assertIn("cursor-based replay", skill)
        self.assertIn("functions.wait", skill)
        self.assertIn("yield_time_ms=604800000", skill)
        self.assertIn("actually killed or lost", skill)
        self.assertIn("never start duplicate pending waits", skill)
        self.assertIn("last action of every turn", skill)
        self.assertIn("unless the user explicitly asks to stop watching", skill)
        self.assertIn("If asked to stop watching, do not re-arm", skill)
        self.assertIn("leave the Job running unless separately asked to cancel it", skill)
        self.assertNotIn("codex queue", skill)
        self.assertNotIn("finish the foreground turn", skill)
        with tempfile.TemporaryDirectory() as directory:
            result = install_skill("claude", home=directory)
            self.assertEqual({p.name for p in result.path.iterdir()}, {"SKILL.md", METADATA_FILE})

    def test_skill_mode_can_switch_between_legacy_and_mcp(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = install_skill("codex", home=directory, mode="legacy")
            marker_path = legacy.path / METADATA_FILE
            legacy_marker = json.loads(marker_path.read_text(encoding="utf-8"))

            self.assertEqual(legacy.status, "installed")
            self.assertEqual(legacy_marker["mode"], "legacy")
            self.assertEqual(
                legacy_marker["bundle_version"], LEGACY_SKILL_BUNDLE_VERSION
            )
            self.assertIn(
                "falcon logs JOB",
                (legacy.path / "SKILL.md").read_text(encoding="utf-8"),
            )

            mcp = install_skill("codex", home=directory, mode="mcp")
            mcp_marker = json.loads(marker_path.read_text(encoding="utf-8"))

            self.assertEqual(mcp.status, "updated")
            self.assertEqual(mcp_marker["mode"], "mcp")
            self.assertEqual(mcp_marker["bundle_version"], SKILL_BUNDLE_VERSION)
            self.assertIn(
                "plan_job",
                (mcp.path / "SKILL.md").read_text(encoding="utf-8"),
            )

    def test_refresh_installed_skills_preserves_each_agent_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            install_skill("codex", home=directory, mode="legacy")
            install_skill("claude", home=directory, mode="mcp")

            self.assertEqual(
                installed_skill_mode("codex", home=directory), "legacy"
            )
            self.assertEqual(
                installed_skill_mode("claude", home=directory), "mcp"
            )
            refreshed = refresh_installed_skills(home=directory)

            self.assertEqual(
                [(operation.agent, operation.status) for operation in refreshed],
                [("codex", "unchanged"), ("claude", "unchanged")],
            )
            self.assertIn(
                "falcon logs JOB",
                (agent_skill_path("codex", home=directory) / "SKILL.md")
                .read_text(encoding="utf-8"),
            )
            self.assertIn(
                "plan_job",
                (agent_skill_path("claude", home=directory) / "SKILL.md")
                .read_text(encoding="utf-8"),
            )

    def test_unknown_skill_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported Falcon skill mode"):
                install_skill("codex", home=directory, mode="unknown")

    def test_official_agent_paths_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertEqual(
                agent_skill_path("codex", home=home),
                home / ".agents" / "skills" / "falcon",
            )
            self.assertEqual(
                agent_skill_path("claude", home=home),
                home / ".claude" / "skills" / "falcon",
            )
            self.assertEqual(
                agent_skill_path("opencode", home=home),
                home / ".config" / "opencode" / "skills" / "falcon",
            )

            results = install_skills("claude,codex,claude", home=home)
            self.assertEqual([result.agent for result in results], ["claude", "codex"])
            self.assertTrue(agent_skill_path("claude", home=home).is_dir())
            self.assertTrue(agent_skill_path("codex", home=home).is_dir())
            self.assertFalse(agent_skill_path("opencode", home=home).exists())

    def test_invalid_agent_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported coding agent"):
                install_skills(["codex", "unknown"], home=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_detection_uses_executables_and_config_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".claude").mkdir()

            def which(command):
                return "/tools/opencode" if command == "opencode" else None

            self.assertEqual(
                detect_agents(home=home, which=which),
                ["claude", "opencode"],
            )
            results = install_skills(home=home, which=which)
            self.assertEqual([result.agent for result in results], ["claude", "opencode"])

    def test_repeated_install_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            first = install_skill("codex", home=directory)
            marker_before = (first.path / METADATA_FILE).read_bytes()
            skill_before = (first.path / "SKILL.md").read_bytes()

            second = install_skill("codex", home=directory)

            self.assertEqual(first.status, "installed")
            self.assertEqual(second.status, "unchanged")
            self.assertFalse(second.changed)
            self.assertEqual((first.path / METADATA_FILE).read_bytes(), marker_before)
            self.assertEqual((first.path / "SKILL.md").read_bytes(), skill_before)
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in first.path.iterdir())
            )

    def test_unchanged_managed_copy_updates_to_new_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            result = install_skill("claude", home=directory)
            marker_path = result.path / METADATA_FILE
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            old_skill = b"---\nname: falcon\ndescription: old\n---\nOld instructions.\n"
            (result.path / "SKILL.md").write_bytes(old_skill)
            marker["bundle_version"] = 0
            marker["files"]["SKILL.md"] = sha256(old_skill)
            marker["source_hash"] = bundle_digest(marker["files"])
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            updated = install_skill("claude", home=directory)
            new_marker = json.loads(marker_path.read_text(encoding="utf-8"))

            self.assertEqual(updated.status, "updated")
            self.assertIn(
                "plan_job",
                (result.path / "SKILL.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(new_marker["bundle_version"], SKILL_BUNDLE_VERSION)
            self.assertNotEqual(new_marker["source_hash"], marker["source_hash"])

    def test_managed_update_removes_obsolete_reference_file(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("codex", home=directory)
            marker_path = installed.path / METADATA_FILE
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            reference = installed.path / "reference.md"
            old_reference = b"old managed reference\n"
            reference.write_bytes(old_reference)
            marker["files"]["reference.md"] = sha256(old_reference)
            marker["bundle_version"] = 0
            marker["source_hash"] = bundle_digest(marker["files"])
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            result = install_skill("codex", home=directory)

            self.assertEqual(result.status, "updated")
            self.assertFalse(reference.exists())

    def test_user_modified_managed_copy_is_never_overwritten_or_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("opencode", home=directory)
            skill_path = installed.path / "SKILL.md"
            skill_path.write_text("my custom Falcon instructions\n", encoding="utf-8")

            repeated = install_skill("opencode", home=directory)
            removed = uninstall_skill("opencode", home=directory)

            self.assertEqual(repeated.status, "conflict")
            self.assertIn("modified", repeated.detail)
            self.assertEqual(removed.status, "conflict")
            self.assertEqual(
                skill_path.read_text(encoding="utf-8"),
                "my custom Falcon instructions\n",
            )
            self.assertTrue((installed.path / METADATA_FILE).exists())

    def test_forced_install_refreshes_modified_managed_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("opencode", home=directory)
            skill_path = installed.path / "SKILL.md"
            skill_path.write_text("my custom Falcon instructions\n", encoding="utf-8")

            refreshed = install_skill("opencode", home=directory, force=True)
            packaged = resources.files("falcon.skills").joinpath("falcon/SKILL.md")

            self.assertEqual(refreshed.status, "updated")
            self.assertEqual(skill_path.read_bytes(), packaged.read_bytes())

    def test_forced_install_still_preserves_unmanaged_skill_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = agent_skill_path("codex", home=directory)
            target.mkdir(parents=True)
            skill = target / "SKILL.md"
            skill.write_text("user-owned\n", encoding="utf-8")

            forced = install_skill("codex", home=directory, force=True)

            self.assertEqual(forced.status, "conflict")
            self.assertEqual(skill.read_text(encoding="utf-8"), "user-owned\n")

    def test_unmanaged_existing_skill_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = agent_skill_path("codex", home=directory)
            target.mkdir(parents=True)
            skill = target / "SKILL.md"
            skill.write_text("user-owned\n", encoding="utf-8")

            installed = install_skill("codex", home=directory)
            removed = uninstall_skill("codex", home=directory)

            self.assertEqual(installed.status, "conflict")
            self.assertEqual(removed.status, "unmanaged")
            self.assertEqual(skill.read_text(encoding="utf-8"), "user-owned\n")
            self.assertFalse((target / METADATA_FILE).exists())

    def test_owned_uninstall_removes_only_managed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("claude", home=directory)
            note = installed.path / "notes.txt"
            note.write_text("keep me\n", encoding="utf-8")

            removed = uninstall_skill("claude", home=directory)

            self.assertEqual(removed.status, "removed")
            self.assertFalse((installed.path / "SKILL.md").exists())
            self.assertFalse((installed.path / METADATA_FILE).exists())
            self.assertEqual(note.read_text(encoding="utf-8"), "keep me\n")
            self.assertEqual(uninstall_skill("claude", home=directory).status, "unmanaged")

    def test_owned_uninstall_removes_empty_directory_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("codex", home=directory)
            removed = uninstall_skill("codex", home=directory)
            repeated = uninstall_skill("codex", home=directory)

            self.assertEqual(removed.status, "removed")
            self.assertFalse(installed.path.exists())
            self.assertEqual(repeated.status, "not-installed")

    def test_corrupt_or_wrong_ownership_metadata_is_a_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("codex", home=directory)
            marker = installed.path / METADATA_FILE
            marker.write_text('{"owner": "someone-else"}\n', encoding="utf-8")

            self.assertEqual(install_skill("codex", home=directory).status, "conflict")
            self.assertEqual(uninstall_skill("codex", home=directory).status, "conflict")
            self.assertTrue(installed.path.exists())

    def test_batch_uninstall_honors_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            install_skills(SUPPORTED_AGENTS, home=directory)

            results = uninstall_skills(["opencode", "codex"], home=directory)

            self.assertEqual([result.agent for result in results], ["opencode", "codex"])
            self.assertFalse(agent_skill_path("opencode", home=directory).exists())
            self.assertFalse(agent_skill_path("codex", home=directory).exists())
            self.assertTrue(agent_skill_path("claude", home=directory).exists())


if __name__ == "__main__":
    unittest.main()
