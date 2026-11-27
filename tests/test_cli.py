"""Проверка установленного CLI на локальном хранилище с синтетической заметкой."""
import json
import os
import subprocess
import sys


def test_index_and_query(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "archive.md").write_text("# Архив\nРезервные копии хранятся 45 дней.", encoding="utf-8")
    env = {
        **os.environ,
        "HARNESS_HOME": str(tmp_path),
        "VAULT_DIR": str(vault),
        "TWINCTX_DB": str(tmp_path / "index.sqlite"),
        "TWINCTX_STORE": "sqlite",
        "TWINCTX_EMBED": "none",
    }

    def run(*args):
        output = subprocess.check_output([sys.executable, "-m", "twinctx", *args], env=env, cwd=tmp_path)
        return json.loads(output)

    run("index")
    hit = run("query", "Сколько дней хранятся резервные копии?", "--json")
    assert "45 дней" in hit["packed"]
    assert hit["manifest"]["items"][0]["source"] == "vault/archive.md"
    assert hit["manifest"]["tokens"] <= hit["manifest"]["budget"]

    miss = run("query", "Орбитальная механика Сатурна", "--json")
    assert miss["packed"] == ""
    assert miss["manifest"]["nothing_relevant"]
