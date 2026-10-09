from click.testing import CliRunner

import frappe_mcp
from frappe_mcp.cli import run


def test_version_prints_package_version():
    result = CliRunner().invoke(run, ['version'])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == frappe_mcp.__version__
