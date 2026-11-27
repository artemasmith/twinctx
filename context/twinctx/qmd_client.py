"""Bounded requests to the TwinCtx QMD sidecar on loopback."""

import json
import urllib.parse
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def hits(path, params, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    url = f"http://127.0.0.1:26358{path}?" + urllib.parse.urlencode(params)
    with opener.open(url, timeout=timeout) as response:
        raw = response.read(32769)
    if len(raw) > 32768:
        raise ValueError("oversized QMD reply")
    result = json.loads(raw)["hits"]
    if not isinstance(result, list) or len(result) > 5:
        raise ValueError("invalid QMD hits")
    return result
