import http.server
import signal
import socketserver
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_agent import create_browser_agent
from sandbox_manager import get_global_manager

load_dotenv()

_http_server: Optional[socketserver.TCPServer] = None
_http_port: int = 8080


def start_http_server() -> Optional[int]:
    global _http_server
    if _http_server is not None:
        return _http_port

    try:
        current_dir = Path(__file__).parent.absolute()

        class VNCRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(current_dir), **kwargs)

            def log_message(self, format, *args):
                # disable log
                pass

        for port in range(_http_port, _http_port + 10):
            try:
                server = socketserver.TCPServer(("", port), VNCRequestHandler)
                server.allow_reuse_address = True

                def run_server():
                    server.serve_forever()

                # run server in background
                thread = threading.Thread(target=run_server, daemon=True)
                thread.start()
                _http_server = server
                return port
            except OSError:
                print(f"create http server at port {port} failed, try next port")
                continue
        return None
    except Exception as e:
        print(f"start http server failed: {str(e)}")


def open_vnc_viewer(vnc_url: str):
    if not vnc_url:
        return

    try:
        current_dir = Path(__file__).parent.absolute()
        vnc_html_path = current_dir / "public" / "vnc.html"
        if not vnc_html_path.exists():
            print(f"file vnc.html is not exist: {vnc_html_path}")
            return

        port = start_http_server()
        encoded_url = urllib.parse.quote(vnc_url, safe="")
        if port:
            http_url = f"http://localhost:{port}/vnc.html?url={encoded_url}"
            print("opening VNC...")
            print(f"HTTP server run at: http://localhost:{port}")
            print(f"VNC URL: {vnc_url[:80]}...")
            print(f"full URL: {http_url[:100]}...")
            webbrowser.open(http_url)
            print("VNC is opened")
            return

        print("HTTP server started failed, try to use file protocal...")
        file_url = f"file://{vnc_html_path}?url={encoded_url}"
        webbrowser.open(file_url)
        print("VNC is opened (file protocal)")
    except Exception as e:
        print(f"open VNC failed: {str(e)}")


def cleanup_sandbox():
    try:
        manager = get_global_manager()
        if manager.is_active():
            print("\n" + "=" * 60)
            print("clearup sandbox...")
            print("=" * 60)
            result = manager.destroy()
            print(f"clearup result: {result}")
        else:
            print("no active sandbox")
    except Exception as e:
        print(f"clearup sandbox failed: {str(e)}")


def signal_handler(signum, frame):
    _, _ = signum, frame
    print("get interrupt system signal")
    cleanup_sandbox()
    print("clearup finished")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("\n" + "=" * 60)
    print("LangChain + AgentRun Browser Sandbox Demo")
    print("=" * 60)
    print()

    try:
        print("init LangChain Agent...")
        agent = create_browser_agent()
        print("Agent init finished")

        queries = [
            "create a browser sandbox",
            "get sandbox info, including VNC URL",
            "navigate to https://www.aliyun.com",
            "screenshot current page",
        ]

        for i, query in enumerate(queries):
            print(f"agent exec query {i}: {query}")
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                output = (
                    result.get("messages", [])[-1].content
                    if isinstance(result.get("messages"), list)
                    else result.get("output", str(result))
                )
                print(f"\noutput:\n{output}\n")

                # open vnc when create sandbox
                if i == 1:
                    try:
                        time.sleep(1)  # wait for sandbox created
                        manager = get_global_manager()
                        if manager.is_active():
                            info = manager.get_info()
                            vnc_url = info.get("vnc_url")
                            if vnc_url:
                                print(f"get VNC URL: {vnc_url[:80]}...")
                                open_vnc_viewer(vnc_url)
                            else:
                                print("cannot get VNC URL, pls check sandbox instance")
                    except Exception as e:
                        print(f"open VNC: {str(e)}")
                        traceback.print_exc()
                # print vnc info
                elif i == 2:
                    try:
                        manager = get_global_manager()
                        if manager.is_active():
                            info = manager.get_info()
                            if info.get("vnc_url"):
                                print("VNC URL:", info.get("vnc_url"))
                    except:
                        pass
            except Exception as e:
                print(f"agent exec query: {str(e)}\n")
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("interactive mode (input 'quit' / 'exit', or ctrl+c . ctrl+d)")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("pls input query: ").strip()
            except KeyboardInterrupt:
                print("\n\nget system signal ctrl+c")
                break
            except EOFError:
                print("\n\nget system signal ctrl+d")
                break
            finally:
                print("clearup...")
                cleanup_sandbox()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\nBye")
                cleanup_sandbox()
                break

            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
                output = (
                    result.get("messages", [])[-1].content
                    if isinstance(result.get("messages"), list)
                    else result.get("output", str(result))
                )
                print(f"\nagent exec query result:\n{output}\n")
            except Exception as e:
                print(f"agent exec query: {str(e)}\n")
                traceback.print_exc()
        cleanup_sandbox()
    except KeyboardInterrupt:
        print("get interrupt system signal (ctrl+c)")
        sys.exit(0)
    except EOFError:
        print("get eof system signal (ctrl+d)")
        sys.exit(0)
    except ValueError as e:
        print(f"env config error: {str(e)}")
    except Exception as e:
        print(f"error: {str(e)}")
        traceback.print_exc()
    finally:
        cleanup_sandbox()


if __name__ == "__main__":
    main()
