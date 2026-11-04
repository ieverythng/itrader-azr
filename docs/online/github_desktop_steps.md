# Applying the `roman/itrader-final` branch with GitHub Desktop, without pushing

The branch exists only on your Mac, in `~/Documents/GitHub/itrader-azr`. Nothing has been pushed.
These steps let you look at it, test it, and decide what to do, while pushing nothing.

## 1. See the branch

1. Open GitHub Desktop. Current Repository: `itrader-azr`.
2. Click the **Current Branch** dropdown. Under "Recent branches" or the full list, pick **`roman/itrader-final`**.
3. The History tab shows the commits made tonight, each prefixed `online:`. Every file in them is new:
   `itrader/online/**`, `itrader/tests/test_online_*.py`, `docs/online/**`. No existing file is touched.

Do not click **Publish branch** or **Push origin**. The repo is public; the agreement was no push.

## 2. Check that nothing existing changed

In Terminal, in the repo folder:

```bash
git diff --stat origin/agent/Watson-dev roman/itrader-final -- . ':!itrader/online' ':!itrader/tests/test_online_*' ':!docs/online'
```

An empty result means the branch adds files and changes nothing else.

## 3. Run it

```bash
cd ~/Documents/GitHub/itrader-azr && bash itrader/online/ops/setup.sh --no-data
```

The data store from tonight is already in `data/online/` (it is ignored by git). To refresh it:

```bash
.venv/bin/python -m itrader.online sync
```

Then `.venv/bin/python -m itrader.online frame` for one framing pass, `paper --once` for one paper
cycle, `dashboard` to render `data/online/runs/dashboard.html`.

## 4. If the CTO wants it

Give him the zip (`itrader_online_branch.zip` next to the handover) or the patch file in it:

```bash
git am < itrader_online_branch/0002-online-full.patch
```

applied on top of `agent/Watson-dev` recreates the branch exactly. He decides where it goes; you push nothing.

## 5. If you want it gone

GitHub Desktop: Branch menu, Delete, choose `roman/itrader-final`, leave "delete on remote" unticked
(there is no remote copy). Your `main` and his branches are untouched either way.
