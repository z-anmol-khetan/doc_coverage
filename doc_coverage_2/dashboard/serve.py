from __future__ import annotations

import argparse
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve doc_coverage_2 dashboard")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    source = Path(args.results).resolve()
    target = ROOT / "results.json"
    shutil.copyfile(source, target)

    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Serving dashboard at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
