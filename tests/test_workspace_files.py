"""The workspace's own file helpers."""

from pyCamSet.workflow.workspace import delete_file


def test_delete_file_removes_a_file_and_ignores_one_already_gone(tmp_path):
    target = tmp_path / "stale.json"
    target.write_text("{}", encoding="utf-8")
    delete_file(target)
    assert not target.exists()
    delete_file(target)  # already gone: not an error
