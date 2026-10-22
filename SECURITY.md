# Security

This file is about **this fork's tree** (strata-fabric). The fork's maintainer owns the response to security
problems here and is the first point of contact; upstream ([Niko1221/Strata](https://github.com/Niko1221/Strata))
is informed through the channel described below, never instead of it.

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**, through GitHub's private
vulnerability reporting **on this fork's repository**: the **Security** tab, **Report a vulnerability**
([direct link](https://github.com/bluenekozkm/strata-fabric/security/advisories/new)). Say what you found, how to
reproduce it (the request, the config keys involved, the engine version and the commit you run), and what an
attacker gains. The fork's maintainer answers there, fixes it in this fork, and credits you in the advisory unless
you ask not to.

If the problem is in code this fork shares with upstream, it almost certainly affects upstream too. Once a fix
exists here, mentioning it upstream is appreciated — but this fork does not forward your report, does not speak
for you, and does not make upstream's response a condition for fixing its own tree.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/bluenekozkm/strata-fabric/issues) in this fork, or as a pull request.

Supported: the current head of this fork's branch. There is no release line; fixes land as new commits here, and
nothing is backported.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`). Out of the box it is reachable from
this PC only.

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in `strata-<model>.json`) opens it to
  your network, and the server then warns when no API key is set.
- **API key.** `"api_key"` in the run config (or `STRATA_API_KEY`, or `--api-key`) is required on `/v1/*` and on
  every endpoint that shows the model's state, requests or answers (`/status`, `/metrics`, `/settings`, `/mcp`,
  `/props`, `/slots`, `/api/requests`, `/config`) and on every `POST`. It is compared in constant time. Set one
  before you open the server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding).** Without an API key the server answers only requests whose `Host` is a name it
  knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's
  name). Others get 403. `"allowed_hosts"` (or `STRATA_ALLOWED_HOSTS`) adds names; with an API key the key decides.
- **Origin check.** Without an API key, a browser `POST` to `/v1/*` that carries an `Origin` header from a page
  elsewhere gets 403 unless the origin is `localhost`, an allowed host, or listed in `"trusted_origins"` /
  `"cors_origins"` — and the body must then be JSON, so a cross-origin page cannot burn GPU time with a form POST.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **MCP tools** are opt-in: only the servers you put in the run config (`"mcp_servers"`, or `--mcp-config` in
  Claude Desktop's format). They run with your user's rights, and the model decides when to call them. Setup:
  [docs/DETAILS.md](docs/DETAILS.md#tools-from-mcp-servers).
- **The request monitor** (the `/api/requests` endpoint, which keeps the last prompts and answers in memory) is
  off unless `"api_monitor": true` (or `--api-monitor`) is set. Apart from it, the server keeps in its process
  memory the text and tokens of its last few prompts, whatever `"api_monitor"` says, to encode the next turn
  faster; no endpoint shows them, and they are gone when the server stops.

## What this fork does not have

The upstream surfaces this fork removed take their exposure with them: there is no desktop app or host process, no
Terminal, no `curl | sh` / `irm | iex` one-line installer, no signed update feed, no built-in web pages. What
remains that reaches the outside world is short:

- **The server above** — loopback by default, key-gated past it.
- **The model downloads** — `prepare.py` fetches each model over HTTPS with a pinned revision, size and SHA-256,
  and a file already checked keeps its mark and is never fetched again. There is no update feed and nothing
  downloads by itself; serving a model never touches the network.

## Audit

No outside security audit has covered Strata, upstream or this fork.
