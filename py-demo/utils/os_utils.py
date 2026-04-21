import subprocess


def shell_exec(command: str) -> str:
    result = None
    try:
        # if check=ture, it will raise subprocess.CalledProcessError when run command failed
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out after 30s"
    except Exception as e:
        return f"[error] {e}"


def run_shell_cmd(command: str) -> int:
    with subprocess.Popen(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as p:
        if p.stdout:
            for line in p.stdout:
                print(line, end="")
        p.wait(timeout=15)
        return p.returncode


if __name__ == "__main__":
    pass
