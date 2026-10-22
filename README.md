# strata-fabric

An independent fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), stripped to its core: the
inference engine and its API server, nothing else. It runs the Qwen3.8-Flash-Next mixture-of-experts model
(and its Coder, Swift 1.5, Unsloth and Gyro variants) on a normal PC: one NVIDIA or AMD graphics card plus
system RAM.

There is no chat app here, no terminal UI, no installer and no update feed. The engine is a binary; the
server is one Python file with an OpenAI- and Anthropic-compatible API; a small `prepare.py` fetches a
model and writes its run config.

## What is in here

- `src/`, `include/`, `CMakeLists.txt`, `cmake/` — the C++ engine (CUDA and HIP backends; SYCL and Metal
  were removed). How it works, and every measured number: [docs/DETAILS.md](docs/DETAILS.md) and the
  [paper](docs/paper/Strata-Paper.pdf). AMD specifics: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU:
  [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- `serve/` — the API server (`serve/server.py`, Python stdlib only) and its tests. It spawns the engine as
  a resident process and speaks a line protocol over pipes (`GEN <max_new> <ids>` in, `T <id>`/`DONE` out;
  batch variants `BGEN`/`BT`/`BDONE`), documented in DETAILS.md.
- `strata_control/` — the shared library layer the server uses for engine components, the vision reader
  and process management.
- `tools/` — the model tools (`strata_tokenizer.py`, `iq_pack.py`, `mtp_pack.py`, the NVFP4 packers, the
  tokenizer's reference implementation) and the benchmark harness (`strata_bench.py`).
- `prepare.py` — download a pinned model, build its tokenizer pack and MTP draft, write the run config,
  and compile the engine.
- `bench/` — micro-benchmarks and measurement results; **local only, not in the repository**
  (`.gitignore`), so measurement data stays on this machine.

## Using it

Python 3.10+ with `pip install -r requirements.txt` (numpy, jinja2, regex, pyyaml, psutil, requests,
pillow). Build the engine once (needs CMake + Ninja, a CUDA toolkit or ROCm):

```sh
python3 prepare.py --build hip                 # or --build cuda; --arch gfx1201 to pin the arch
```

Get a model (pins revision, size and SHA-256; resumable; lands in `../Strata-data` next to this folder):

```sh
python3 prepare.py --list                      # the catalog
python3 prepare.py --model Gyro-M              # + --low-ram to write the experts into one file
```

Serve it:

```sh
python3 -m serve.server --engine strata --config strata-gyro-gyro-m.json --port 8080
```

The server binds to 127.0.0.1 and, past loopback, refuses to start without an API key. Endpoints:
`/v1/chat/completions`, `/v1/messages` (+ `count_tokens`), `/v1/responses`, `/v1/models`, `/v1/images/generations`,
`/props`, `/slots`, `/health`, `/metrics` (+ Prometheus), `/v1/telemetry`, and an MCP client hub for the
model's own tool calls (`--mcp-config`). There is no built-in web page; any OpenAI/Anthropic client works.

## Upstream policy

This fork keeps upstream as a **read-only source of engine updates**: merge from it, never push to it.

```sh
git fetch upstream && git merge upstream/main
```

The cleanup commit removed (among others): `apps/`, `strata_cli/`, `bin/`, `strata.py`, `setup.py`,
`setup.sh`, `packaging/`, `sycl/`, `metal/`, `serve/web/`, `docker/`, `integrations/`, `.github/` bots,
`tests/{terminal,tuning,time_skip,desktop,sycl}`, the NVFP4 model sizes from the catalog, the Terminal and
app docs, and most of `tools/` (the keep-list is in that commit); `bench/` is untracked (local only). After
a merge, resolve each
modify/delete conflict on those paths with `git rm`, then run the serve suite:

```sh
python3 -m unittest discover -s serve -p 'test_*.py'
```

The code keeps the name `strata` on purpose — the binary, the modules and the CMake project — so upstream
merges stay cheap. Only the project identity is strata-fabric.

## Tests

```sh
python3 -m unittest discover -s serve -p 'test_*.py'    # the server and the API, no GPU needed (mock engine)
cmake -B build-test -DSTRATA_BUILD_TESTS=ON && cmake --build build-test   # the engine's own tests
```

## License

MIT, like upstream — see [LICENSE](LICENSE). The models have their own licenses (Qwen Community License
1.0 and the fine-tunes' terms); `prepare.py --list` prints what applies.
