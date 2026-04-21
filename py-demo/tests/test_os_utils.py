from utils import run_shell_cmd, shell_exec


def test_shell_exec():
    cmd = "ls -l /tmp/test"
    output = shell_exec(cmd)
    print(f"exec [{cmd}]:\n{output}")


def test_run_shell_cmd():
    cmd = "ls -l /tmp/test"
    ret_code = run_shell_cmd(cmd)
    assert ret_code == 0, f"run command [{cmd}] with non-zero returned code: {ret_code}"
