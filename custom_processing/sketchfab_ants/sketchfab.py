"""Listing and downloading a Sketchfab user's models through the Sketchfab Data API v3.

Listing is public. Downloading needs an API token (sketchfab.com/settings/password, "API Token"), read from
the environment variable SKETCHFAB_TOKEN or from a token file (default ~/.sketchfab_token). Only models the
author marked downloadable can be fetched; their licence still applies (tunosemi's are CC BY-NC-ND / CC BY-ND,
used for Holotype with the author's permission).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

API = "https://api.sketchfab.com/v3"
UA = "model-list-client"


def _get_json(url, token=None, tries=6):
    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = f"Token {token}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(30 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def list_user_models(username, pause=0.3):
    """All public models of a user, deduplicated by uid.

    The search endpoint's cursor pagination returns duplicates (and therefore misses models) under its default
    relevance order, so several sort orders are paged through and merged.
    """
    models = {}
    for sort in ("", "-publishedAt", "publishedAt", "-likeCount", "-viewCount"):
        url = f"{API}/search?type=models&user={username}&count=24" + (f"&sort_by={sort}" if sort else "")
        while url:
            d = _get_json(url)
            for m in d["results"]:
                models.setdefault(m["uid"], m)
            url = d.get("next")
            time.sleep(pause)
    rows = []
    for m in models.values():
        lic = m.get("license")
        rows.append(dict(uid=m["uid"], name=m["name"], downloadable=bool(m.get("isDownloadable")),
                         verts=m.get("vertexCount"), faces=m.get("faceCount"),
                         license=lic.get("label") if isinstance(lic, dict) else lic,
                         tags=[t["name"] for t in m.get("tags", [])], published=m.get("publishedAt")))
    return sorted(rows, key=lambda r: r["name"].lower())


def model_key(name, uid):
    """Folder/file stem for a model: sanitised title + first 8 characters of the uid."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.replace("Ｍ", "M")).strip("_")
    return f"{s}_{uid[:8]}"


def read_token(token_file=None):
    tok = os.environ.get("SKETCHFAB_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.expanduser(token_file or "~/.sketchfab_token")
    if not os.path.exists(path):
        raise SystemExit(f"No Sketchfab API token: set SKETCHFAB_TOKEN or write it to {path}")
    return open(path).read().strip()


def _fetch(url, dest):
    tmp = dest + ".part"  # pre-signed storage URL: no auth header
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r, \
            open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def download_models(models, raw_root, token, fmt="glb", pause=2.0):
    """Downloads each downloadable model to <raw_root>/<key>/<key>.<fmt>; skips models already downloaded.

    Every model gets its own new directory. Progress is kept in <raw_root>/download_manifest.json, so an
    interrupted run can be restarted.
    """
    os.makedirs(raw_root, exist_ok=True)
    manifest_path = os.path.join(raw_root, "download_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    todo = [m for m in models if m["downloadable"]]
    for i, m in enumerate(todo):
        uid, key = m["uid"], model_key(m["name"], m["uid"])
        dest = os.path.join(raw_root, key, f"{key}.{fmt}")
        if os.path.exists(dest) and manifest.get(uid, {}).get("status") == "ok":
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        entry = dict(name=m["name"], uid=uid, key=key, license=m.get("license"), author="tunosemi",
                     url=f"https://sketchfab.com/3d-models/{uid}", format=fmt)
        try:
            info = _get_json(f"{API}/models/{uid}/download", token)
            if fmt not in info:
                entry.update(status="no_format", available=sorted(info))
            else:
                _fetch(info[fmt]["url"], dest)
                with open(dest, "rb") as f:
                    ok = f.read(4) == b"glTF" if fmt == "glb" else True
                entry.update(status="ok" if ok else "bad_file", bytes=os.path.getsize(dest), path=dest)
        except Exception as e:  # keep going; the manifest records the failure
            entry.update(status="error", error=repr(e))
        manifest[uid] = entry
        json.dump(manifest, open(manifest_path, "w"), indent=1, ensure_ascii=False)
        print(f"[{i + 1}/{len(todo)}] {m['name']}: {entry['status']}", flush=True)
        time.sleep(pause)
    counts = {}
    for e in manifest.values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return counts
