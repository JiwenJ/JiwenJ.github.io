/* Shared article counters use Waline's article API; browser storage is only a like marker. */
(() => {
  "use strict";

  const script = document.currentScript;
  const server = script && script.dataset.engagementServer;
  const roots = [...document.querySelectorAll("[data-engagement-path]")];
  if (!server || !roots.length) return;

  let endpoint;
  try {
    endpoint = new URL(`${server.replace(/\/+$/, "")}/api/article`);
    if (!/^https?:$/.test(endpoint.protocol)) return;
  } catch (_) {
    return;
  }

  const format = new Intl.NumberFormat(document.documentElement.lang || "zh-CN");
  const states = new Map();
  const storageKey = (path) => `jiwen:article-like:v1:${path}`;
  const readMarker = (path) => {
    try { return localStorage.getItem(storageKey(path)) === "1"; }
    catch (_) { return false; }
  };

  roots.forEach((root) => {
    const path = root.dataset.engagementPath;
    if (!states.has(path)) {
      states.set(path, { path, roots: [], views: null, likes: null, liked: readMarker(path), ready: false, busy: false });
    }
    states.get(path).roots.push(root);
  });

  const count = (value) => {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Invalid counter response");
    return value;
  };

  const render = (state) => {
    state.roots.forEach((root) => {
      root.querySelectorAll("[data-view-count]").forEach((el) => {
        el.textContent = state.views === null ? "—" : format.format(state.views);
      });
      root.querySelectorAll("[data-like-count]").forEach((el) => {
        el.textContent = state.likes === null ? "—" : format.format(state.likes);
      });
      root.querySelectorAll("[data-like-button]").forEach((button) => {
        button.disabled = !state.ready || state.busy;
        button.setAttribute("aria-pressed", String(state.liked));
        button.setAttribute("aria-label", state.liked ? "取消对这篇文章的点赞" : "为这篇文章点赞");
        button.setAttribute("aria-busy", String(state.busy));
        button.querySelectorAll("[data-like-label]").forEach((el) => { el.textContent = state.liked ? "已赞" : "点赞"; });
      });
    });
  };

  const announce = (state, message) => {
    state.roots.forEach((root) => {
      root.querySelectorAll("[data-engagement-status]").forEach((el) => { el.textContent = message; });
    });
  };

  const request = async (url, options = {}) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(url, { ...options, credentials: "omit", signal: controller.signal });
      if (!response.ok) throw new Error("Counter service unavailable");
      const body = await response.json();
      if (body.errno !== 0 || !Array.isArray(body.data)) throw new Error("Invalid counter response");
      return body.data;
    } finally {
      clearTimeout(timeout);
    }
  };

  const change = async (path, type, action = "inc") => {
    const url = new URL(endpoint);
    url.searchParams.set("lang", "zh-CN");
    const rows = await request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, type, action }),
    });
    return count(rows[0] && rows[0][type]);
  };

  states.forEach((state) => {
    state.roots.forEach((root) => {
      root.querySelectorAll("[data-like-button]").forEach((button) => {
        button.addEventListener("click", async () => {
          if (!state.ready || state.busy) return;
          const previous = state.liked;
          state.busy = true;
          announce(state, "");
          render(state);
          try {
            state.likes = await change(state.path, "reaction0", previous ? "desc" : "inc");
            state.liked = !previous;
            try {
              if (state.liked) localStorage.setItem(storageKey(state.path), "1");
              else localStorage.removeItem(storageKey(state.path));
            } catch (_) { /* Counts remain on the server even when local storage is unavailable. */ }
            announce(state, state.liked ? "谢谢你的喜欢。" : "已取消点赞。");
          } catch (_) {
            announce(state, "暂时无法保存，请稍后再试。");
          } finally {
            state.busy = false;
            render(state);
          }
        });
      });
    });
    render(state);
  });

  const initialize = async () => {
    const paths = [...states.keys()];
    const url = new URL(endpoint);
    url.searchParams.set("path", paths.join(","));
    url.searchParams.set("type", "time,reaction0");
    url.searchParams.set("lang", "zh-CN");
    try {
      const rows = await request(url);
      if (rows.length !== paths.length) throw new Error("Incomplete counter response");
      const counters = rows.map((row) => ({ views: count(row.time), likes: count(row.reaction0) }));
      paths.forEach((path, index) => {
        const state = states.get(path);
        Object.assign(state, counters[index], { ready: true });
        if (state.likes === 0) state.liked = false;
        render(state);
      });
    } catch (_) {
      states.forEach((state) => announce(state, "统计暂不可用，请稍后刷新。"));
      return;
    }

    // Only an article page increments its own PV. Lists only read counters.
    for (const state of states.values()) {
      if (!state.roots.some((root) => root.dataset.engagementCurrent === "true")) continue;
      try {
        state.views = await change(state.path, "time");
        render(state);
      } catch (_) { /* The last confirmed server count remains visible. */ }
    }
  };

  initialize();
})();
