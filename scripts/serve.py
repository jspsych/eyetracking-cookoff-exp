#!/usr/bin/env python3
"""
Serve the experiment on localhost, with caching off.

`python3 -m http.server` sends no cache headers at all, which lets Chrome apply its own
heuristics: edit a file in js/, reload, and get the previous version back with no indication
that anything is stale. That wastes an afternoon exactly once. This sends `no-store` on
everything, so a reload is always a reload.

getUserMedia needs a secure context, and localhost counts as one, so this is enough to run the
whole experiment.

    python3 scripts/serve.py [port]
"""

import functools
import http.server
import os
import pathlib
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # One line per request is noise once the 20 MB model and the wasm are in flight.
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    root = pathlib.Path(__file__).resolve().parent.parent
    os.chdir(root)
    handler = functools.partial(NoCacheHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        # flush=True so the banner appears even when stdout is redirected or piped.
        print(f"the cook-off is at http://localhost:{port}/index.html  (ctrl-c to stop)", flush=True)
        print("checks: /test/port-check.html  /test/pipeline-check.html", flush=True)
        print("        /test/trial-check.html  /test/deferred-check.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
