import hashlib
import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from falcon.agent_skills import (
    METADATA_FILE,
    SKILL_BUNDLE_VERSION,
    SUPPORTED_AGENTS,
    agent_skill_path,
    detect_agents,
    install_skill,
    install_skills,
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
    def test_packaged_skill_is_portable_concise_and_progressive(self):
        root = resources.files("falcon.skills").joinpath("falcon")
        skill = root.joinpath("SKILL.md").read_text(encoding="utf-8")
        reference = root.joinpath("reference.md").read_text(encoding="utf-8")
        frontmatter = skill.split("---", 2)[1]

        self.assertIn("name: falcon", frontmatter)
        self.assertIn("description:", frontmatter)
        self.assertNotIn("allowed-tools:", frontmatter)
        self.assertLessEqual(len(skill.splitlines()), 40)
        self.assertEqual(skill.count("- `falcon "), 8)
        self.assertIn("[reference.md](reference.md)", skill)
        self.assertIn("exit codes", skill)
        self.assertIn("falcon get JOB --output json", reference)

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
                "falcon submit --gpu h100",
                (result.path / "SKILL.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(new_marker["bundle_version"], SKILL_BUNDLE_VERSION)
            self.assertNotEqual(new_marker["source_hash"], marker["source_hash"])

    def test_managed_update_does_not_claim_an_untracked_future_file(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = install_skill("codex", home=directory)
            marker_path = installed.path / METADATA_FILE
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["files"].pop("reference.md")
            marker["bundle_version"] = 0
            marker["source_hash"] = bundle_digest(marker["files"])
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            reference = installed.path / "reference.md"
            reference.write_text("my private reference\n", encoding="utf-8")

            result = install_skill("codex", home=directory)

            self.assertEqual(result.status, "conflict")
            self.assertIn("unmanaged reference.md", result.detail)
            self.assertEqual(reference.read_text(encoding="utf-8"), "my private reference\n")

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
            self.assertFalse((installed.path / "reference.md").exists())
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
