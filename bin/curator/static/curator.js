// Curator frontend (generic / schema-driven rewrite).
//
// Loads /api/schema to discover columns + STRUCT subfields, lets the user pick
// any subset via the Columns dialog, fetches /api/rows with the chosen list,
// and renders a dynamic table. Switching parquets via the file picker reloads
// the schema and rebuilds the table.
//
// Backwards-compat with the old curator is intentionally dropped — the legacy
// version is preserved in curator_legacy.js if needed.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ---------- formatting ----------
const fmtInt = (n) => (n ?? 0).toLocaleString("en-US");
const fmtSec = (s) => (s == null ? "—" : `${Number(s).toFixed(1)}s`);
const fmtScore = (s, digits = 2) =>
  (s == null ? "—" : Number(s).toFixed(digits));

function scoreClass(value, goodMin, badMax) {
  if (value == null) return "score-na";
  if (value >= goodMin) return "score-good";
  if (value <= badMax) return "score-bad";
  return "score-warn";
}

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function truncate(s, n = 80) {
  if (s == null) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// ---------- state ----------
const state = {
  parquet: null,           // active parquet path (server-side)
  parquetName: null,       // basename for UI
  schema: [],              // [{name, type, subfields?:[{name, type}]}]
  paths: [],               // flat list of selectable dot-paths
  pathMeta: {},            // path -> {type, parent?, label?, sortable}
  defaultColumns: [],      // from /api/schema
  visiblePaths: [],        // user's choice, ordered
  colWidths: {},           // path -> px width (persisted globally)
  filters: { source: [], q: "" },
  sort: "utterance_id",
  dir: "asc",
  page: 1,
  page_size: 50,
  totalRows: 0,
};

// One global column choice across ALL parquets. Switching parquets just
// filters the global list to whichever paths exist in the new schema —
// see switchParquet(). This was a per-parquet key originally; 2026-05-27
// the user asked for a single shared setting.
const LS_KEY = "curator.v2.cols";
const LS_WIDTHS = "curator.v2.widths";  // global path → px width

// Default px widths by column kind. Used when the user hasn't set one yet.
const DEFAULT_COL_WIDTHS = {
  _audio_player: 360,
  utterance_id: 260,
  source: 140,
  duration_sec: 70,
  audio_path: 280,
  refined_audio_path: 200,
  "transcripts.source_caption": 280,
  "transcripts.source_caption_normalized": 280,
};
const DEFAULT_COL_WIDTH_FALLBACK = 140;

function defaultWidthFor(path) {
  if (DEFAULT_COL_WIDTHS[path]) return DEFAULT_COL_WIDTHS[path];
  // Heuristics by suffix: long string columns get wider
  if (path.startsWith("transcripts.")) return 260;
  if (path.endsWith("_path") || path.endsWith("audio_path")) return 240;
  return DEFAULT_COL_WIDTH_FALLBACK;
}

function loadColumnWidths() {
  try {
    const raw = localStorage.getItem(LS_WIDTHS);
    if (raw) {
      const obj = JSON.parse(raw);
      if (obj && typeof obj === "object") return obj;
    }
  } catch {}
  return {};
}

function saveColumnWidths(widths) {
  try {
    localStorage.setItem(LS_WIDTHS, JSON.stringify(widths));
  } catch {}
}

// Path can contain dots/brackets that are invalid in CSS class names
// (e.g. "quality_flags.dnsmos_ovrl"). Sanitize for class use.
function pathToCssClass(path) {
  return "col-" + path.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function loadVisibleColumns(parquetName, fallback) {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length) return arr;
    }
  } catch {}
  return [...fallback];
}

function saveVisibleColumns(parquetName, paths) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(paths));
  } catch {}
}

// ---------- schema utilities ----------
function flattenSchema(schema) {
  // Build the list of selectable column paths and lookup metadata.
  const paths = [];
  const meta = {};
  for (const col of schema) {
    if (col.subfields && col.subfields.length) {
      for (const sf of col.subfields) {
        const path = `${col.name}.${sf.name}`;
        paths.push(path);
        meta[path] = {
          type: sf.type,
          parent: col.name,
          label: path,
          sortable: isSortableType(sf.type),
        };
      }
      // Don't expose the raw STRUCT as a column (would render as JSON);
      // its subfields are what the user wants.
    } else {
      paths.push(col.name);
      meta[col.name] = {
        type: col.type,
        parent: null,
        label: col.name,
        sortable: isSortableType(col.type),
      };
    }
  }
  return { paths, meta };
}

function isSortableType(t) {
  if (!t) return false;
  const u = String(t).toUpperCase();
  return !(u.startsWith("STRUCT(") || u.startsWith("LIST(") ||
           u.startsWith("MAP(") || u.endsWith("[]"));
}

function categoryFor(path) {
  // For grouping in the column dialog: top-level columns -> "top",
  // STRUCT subfields -> the parent struct's name.
  return path.includes(".") ? path.split(".")[0] : "top";
}

// ---------- cell rendering ----------
function renderTranscript(v) {
  if (v == null || v === "") return '<span class="score-na">—</span>';
  return `<div class="transcript-cell">${escapeHtml(v)}</div>`;
}

// Map column-path patterns to specialized renderers. Falls back to a
// type-based generic renderer if no name match.
const NAMED_RENDERERS = {
  utterance_id: (v) =>
    `<a class="utt-link" data-utt="${escapeHtml(v)}" href="#">${escapeHtml(v)}</a>`,
  source: (v) =>
    `<span class="src-badge src-${escapeHtml(v)}">${escapeHtml(v)}</span>`,
  duration_sec: (v) => fmtSec(v),
  "transcripts.source_caption": renderTranscript,
  "transcripts.source_caption_normalized": renderTranscript,
  "quality_flags.dnsmos_ovrl": (v) =>
    `<span class="${scoreClass(v, 3.0, 2.0)}">${fmtScore(v)}</span>`,
  "quality_flags.dnsmos_sig": (v) =>
    `<span class="${scoreClass(v, 3.5, 2.5)}">${fmtScore(v)}</span>`,
  "quality_flags.dnsmos_bak": (v) =>
    `<span class="${scoreClass(v, 3.0, 2.0)}">${fmtScore(v)}</span>`,
  "quality_flags.vad_speech_ratio": (v) =>
    `<span class="${scoreClass(v, 0.7, 0.3)}">${fmtScore(v)}</span>`,
  "quality_flags.whole_clip_top1": (v) => {
    if (v == null) return '<span class="score-na">—</span>';
    return `<span class="${v === "hu" ? "score-good" : "score-bad"}">${escapeHtml(v)}</span>`;
  },
  "quality_flags.whole_clip_hu_prob": (v) => fmtScore(v),
  "quality_flags.smoke_bucket": (v) => {
    if (!v) return '<span class="score-na">—</span>';
    const cls = v === "outlier" ? "score-warn"
              : v === "random" ? "score-na" : "score-good";
    return `<span class="${cls}">${escapeHtml(v)}</span>`;
  },
  "quality_flags.is_clipped": renderBool,
  // Legacy v4 flat-schema parquet aliases (alias keys; same logic):
  qf_dnsmos_ovrl: (v) => `<span class="${scoreClass(v, 3.0, 2.0)}">${fmtScore(v)}</span>`,
  qf_vad_speech_ratio: (v) => `<span class="${scoreClass(v, 0.7, 0.3)}">${fmtScore(v)}</span>`,
  qf_lid: (v) => {
    if (v == null) return '<span class="score-na">—</span>';
    return `<span class="${v === "hu" ? "score-good" : "score-bad"}">${escapeHtml(v)}</span>`;
  },
  qf_is_clipped: renderBool,
};

function renderBool(v) {
  if (v === true)  return '<span class="bool-yes">✓</span>';
  if (v === false) return '<span class="bool-no">×</span>';
  return '<span class="score-na">—</span>';
}

function genericRender(value, type) {
  if (value == null) return '<span class="score-na">—</span>';
  const u = String(type || "").toUpperCase();
  if (u === "BOOLEAN") return renderBool(value);
  if (u === "DOUBLE" || u === "FLOAT") return fmtScore(value);
  if (u.includes("INT")) return escapeHtml(value);
  if (Array.isArray(value)) return escapeHtml(value.join(", "));
  if (typeof value === "object") {
    const s = JSON.stringify(value);
    return `<span title="${escapeHtml(s)}">${escapeHtml(truncate(s, 60))}</span>`;
  }
  // VARCHAR / string: truncate long values
  const s = String(value);
  return s.length > 120
    ? `<span title="${escapeHtml(s)}">${escapeHtml(truncate(s, 120))}</span>`
    : escapeHtml(s);
}

function renderCell(path, row) {
  if (path === "_audio_player") {
    return renderAudioPlayer(row);
  }
  const value = row[path];
  const renderer = NAMED_RENDERERS[path];
  if (renderer) return renderer(value, row);
  const type = (state.pathMeta[path] || {}).type;
  return genericRender(value, type);
}

function renderAudioPlayer(row) {
  const audioPath = row.audio_path;
  const rowIndex = row.parquet_row_index;
  const orig = row._relative_audio_path;
  const refined = row._relative_refined_audio_path;

  // Dispatch based on audio_path format:
  //   .parquet shard + parquet_row_index → /audio_parquet (vp_labeled)
  //   ending in known audio ext → /audio/<rel_path> (everything else)
  let srcUrl = null;
  if (audioPath && audioPath.endsWith(".parquet") && rowIndex != null) {
    srcUrl = `/audio_parquet?path=${encodeURIComponent(audioPath)}&row_index=${rowIndex}`;
  } else if (orig) {
    srcUrl = `/audio/${encodeURI(orig)}`;
  }

  if (!srcUrl && !refined) {
    return '<span class="audio-na">no audio</span>';
  }

  const parts = [];
  if (srcUrl) {
    parts.push(
      '<div class="audio-row"><span class="audio-label">orig</span>' +
      `<audio controls preload="none" src="${srcUrl}"></audio></div>`
    );
  }
  if (refined) {
    parts.push(
      '<div class="audio-row"><span class="audio-label">refined</span>' +
      `<audio controls preload="none" src="/audio/${encodeURI(refined)}"></audio></div>`
    );
  }
  return parts.join("");
}

// ---------- API ----------
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

async function loadParquetList() {
  const j = await api("/api/parquets");
  return j;
}

async function loadSchema(parquet) {
  const url = parquet
    ? `/api/schema?parquet=${encodeURIComponent(parquet)}`
    : "/api/schema";
  return api(url);
}

async function loadStats(parquet) {
  const url = parquet
    ? `/api/stats?parquet=${encodeURIComponent(parquet)}`
    : "/api/stats";
  return api(url);
}

async function loadSources(parquet) {
  const url = parquet
    ? `/api/sources?parquet=${encodeURIComponent(parquet)}`
    : "/api/sources";
  return api(url);
}

async function loadRows() {
  const qs = new URLSearchParams();
  qs.set("parquet", state.parquetName);
  qs.set("page", state.page);
  qs.set("page_size", state.page_size);
  qs.set("sort", state.sort);
  qs.set("dir", state.dir);
  // Include _audio_player as a virtual column (it's not on the server side;
  // we just want the audio path columns fetched, which is automatic).
  const dataPaths = state.visiblePaths.filter(p => p !== "_audio_player");
  if (dataPaths.length) qs.set("columns", dataPaths.join(","));
  for (const s of state.filters.source) {
    qs.append("source", s);
  }
  if (state.filters.q) {
    qs.set("q", state.filters.q);
  }
  for (const [path, range] of Object.entries(state.filters.ranges || {})) {
    if (range.min != null && range.min !== "") {
      qs.append(`filter_min[${path}]`, range.min);
    }
    if (range.max != null && range.max !== "") {
      qs.append(`filter_max[${path}]`, range.max);
    }
  }
  return api(`/api/rows?${qs}`);
}

// ---------- rendering ----------
function renderHeaderMeta(stats) {
  const meta = stats.total_hours != null
    ? `${fmtInt(stats.total_rows)} rows · ${stats.total_hours} h`
    : `${fmtInt(stats.total_rows)} rows`;
  $("#header-meta").textContent =
    `${stats.parquet_name} — ${meta}`;
}

function widthFor(path) {
  return state.colWidths[path] || defaultWidthFor(path);
}

function renderTableHeader() {
  const head = $("#rows-head");
  const cells = state.visiblePaths.map(path => {
    const sortable = path === "_audio_player"
      ? false
      : (state.pathMeta[path] || {}).sortable;
    const sortClass = state.sort === path
      ? (state.dir === "asc" ? "sort-asc" : "sort-desc")
      : "";
    // Header shows just the leaf name (e.g. `dnsmos_ovrl` instead of
    // `quality_flags.dnsmos_ovrl`). Subfield names are unique across
    // STRUCT parents in practice. The full path is preserved as a tooltip
    // (`title` attr) so the user can still see the full lineage on hover.
    const leafLabel = path === "_audio_player" ? "audio"
      : path.includes(".") ? path.split(".").pop()
      : path;
    const cls = pathToCssClass(path);
    const w = widthFor(path);
    const titleAttr = path === "_audio_player" ? "" : ` title="${escapeHtml(path)}"`;
    const sortAttrs = sortable
      ? ` class="${cls} sortable ${sortClass}" data-sort="${escapeHtml(path)}"`
      : ` class="${cls}"`;
    const handle = `<span class="col-resize-handle" data-path="${escapeHtml(path)}" title="Drag to resize"></span>`;
    return `<th${sortAttrs}${titleAttr} style="width:${w}px;min-width:${w}px;max-width:${w}px"><span class="col-label">${escapeHtml(leafLabel)}</span>${handle}</th>`;
  });
  head.innerHTML = `<tr>${cells.join("")}</tr>`;
}

function renderTableRows(data) {
  const body = $("#rows-body");
  const html = data.rows.map(r => {
    const cells = state.visiblePaths.map(path => {
      const cls = pathToCssClass(path);
      const w = widthFor(path);
      return `<td class="${cls}" style="width:${w}px;min-width:${w}px;max-width:${w}px">${renderCell(path, r)}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  body.innerHTML = html || `<tr><td colspan="${state.visiblePaths.length}" class="empty">no rows</td></tr>`;
}

// Drag-to-resize for column headers. Mousedown on the handle starts a
// drag; mousemove updates that column's width; mouseup persists.
let _drag = null;
document.addEventListener("mousedown", (ev) => {
  const handle = ev.target.closest(".col-resize-handle");
  if (!handle) return;
  ev.preventDefault();
  ev.stopPropagation();
  const th = handle.closest("th");
  const path = handle.dataset.path;
  _drag = {
    path,
    th,
    startX: ev.clientX,
    startWidth: th.getBoundingClientRect().width,
  };
  document.body.classList.add("col-resizing");
});
document.addEventListener("mousemove", (ev) => {
  if (!_drag) return;
  const dx = ev.clientX - _drag.startX;
  const newW = Math.max(50, Math.round(_drag.startWidth + dx));
  // Apply to the th and all td.<cls> cells
  const cls = pathToCssClass(_drag.path);
  document.querySelectorAll(`th.${cls}, td.${cls}`).forEach(el => {
    el.style.width = newW + "px";
    el.style.minWidth = newW + "px";
    el.style.maxWidth = newW + "px";
  });
  _drag.lastWidth = newW;
});
document.addEventListener("mouseup", () => {
  if (!_drag) return;
  if (_drag.lastWidth) {
    state.colWidths[_drag.path] = _drag.lastWidth;
    saveColumnWidths(state.colWidths);
  }
  document.body.classList.remove("col-resizing");
  _drag = null;
});

function renderPagination(data) {
  const total = data.total;
  const pages = Math.max(1, Math.ceil(total / state.page_size));
  const cur = state.page;
  const showAt = (p) =>
    `<button class="page-btn ${p === cur ? "active" : ""}" data-page="${p}">${p}</button>`;
  const parts = [];
  parts.push(`<button class="page-btn" data-page="prev" ${cur <= 1 ? "disabled" : ""}>‹</button>`);
  const range = [];
  range.push(1);
  for (let p = cur - 2; p <= cur + 2; p++) {
    if (p > 1 && p < pages) range.push(p);
  }
  if (pages > 1) range.push(pages);
  const uniq = [...new Set(range)].sort((a, b) => a - b);
  let last = 0;
  for (const p of uniq) {
    if (last && p - last > 1) parts.push('<span class="page-gap">…</span>');
    parts.push(showAt(p));
    last = p;
  }
  parts.push(`<button class="page-btn" data-page="next" ${cur >= pages ? "disabled" : ""}>›</button>`);
  parts.push(`<span class="page-info">page ${cur} / ${pages} — ${fmtInt(total)} rows</span>`);
  $("#pagination").innerHTML = parts.join("");
}

function renderResultMeta(data) {
  const from = (data.page - 1) * data.page_size + 1;
  const to = Math.min(data.total, from + data.rows.length - 1);
  $("#result-meta").textContent =
    `Showing ${fmtInt(from)}–${fmtInt(to)} of ${fmtInt(data.total)}`;
}

// ---------- columns dialog ----------
function openColumnsDialog() {
  const container = $("#columns-categories");
  // Group paths by category (top-level / per STRUCT)
  const byCat = {};
  for (const path of state.paths) {
    const cat = categoryFor(path);
    if (!byCat[cat]) byCat[cat] = [];
    byCat[cat].push(path);
  }
  const order = ["top", ...Object.keys(byCat).filter(k => k !== "top").sort()];
  const selected = new Set(state.visiblePaths);

  // For nested STRUCT columns, display the leaf name only (e.g.
  // "dnsmos_ovrl" instead of "quality_flags.dnsmos_ovrl") inside the
  // category section — the category header already shows the parent.
  function leafName(path) {
    return path.includes(".") ? path.split(".").slice(1).join(".") : path;
  }

  const sections = order
    .filter(cat => byCat[cat] && byCat[cat].length)
    .map(cat => {
      const items = byCat[cat].map(path => {
        const checked = selected.has(path) ? "checked" : "";
        const display = cat === "top" ? path : leafName(path);
        return `<label class="col-item" data-search="${escapeHtml(path.toLowerCase())}">
          <input type="checkbox" data-path="${escapeHtml(path)}" ${checked}>
          <span class="col-name">${escapeHtml(display)}</span>
        </label>`;
      }).join("");
      const label = cat === "top" ? "Top-level" : cat;
      const checkedCount = byCat[cat].filter(p => selected.has(p)).length;
      const subActions = `<div class="col-cat-actions">
        <button class="btn-link" data-cat-all="${escapeHtml(cat)}">all</button>
        <button class="btn-link" data-cat-none="${escapeHtml(cat)}">none</button>
      </div>`;
      return `<details class="col-category" open>
        <summary><span class="col-cat-label">${escapeHtml(label)}</span> <span class="col-cat-count">${checkedCount}/${byCat[cat].length}</span> ${subActions}</summary>
        <div class="col-items col-grid">${items}</div>
      </details>`;
    });

  // Synthetic audio player as a pickable "column"
  const audioChecked = selected.has("_audio_player") ? "checked" : "";
  const audioBlock = `<details class="col-category" open>
    <summary><span class="col-cat-label">Audio player (virtual)</span></summary>
    <div class="col-items">
      <label class="col-item" data-search="audio player">
        <input type="checkbox" data-path="_audio_player" ${audioChecked}>
        <span class="col-name">inline audio</span>
      </label>
    </div>
  </details>`;

  container.innerHTML = audioBlock + sections.join("");

  // Wire per-category all/none buttons (event delegation in the
  // container itself — single handler).
  container.onclick = (ev) => {
    const allBtn = ev.target.closest("[data-cat-all]");
    const noneBtn = ev.target.closest("[data-cat-none]");
    if (allBtn || noneBtn) {
      ev.preventDefault();
      const cat = (allBtn || noneBtn).dataset[allBtn ? "catAll" : "catNone"];
      const section = (allBtn || noneBtn).closest(".col-category");
      const boxes = section.querySelectorAll("input[type='checkbox']");
      boxes.forEach(b => { b.checked = !!allBtn; });
      updateCategoryCounts();
    }
  };

  // Live search filter
  const searchInput = $("#columns-search");
  if (searchInput) {
    searchInput.value = "";
    searchInput.oninput = () => {
      const q = searchInput.value.trim().toLowerCase();
      $$("#columns-categories .col-item").forEach(el => {
        const match = !q || (el.dataset.search || "").includes(q);
        el.style.display = match ? "" : "none";
      });
    };
  }

  // Per-category count updates on toggle
  $$("#columns-categories input[type='checkbox']").forEach(b => {
    b.addEventListener("change", updateCategoryCounts);
  });

  $("#columns-modal").showModal();
}

function updateCategoryCounts() {
  $$("#columns-categories details.col-category").forEach(section => {
    const boxes = section.querySelectorAll("input[type='checkbox']");
    const checked = section.querySelectorAll("input[type='checkbox']:checked");
    const counter = section.querySelector(".col-cat-count");
    if (counter) counter.textContent = `${checked.length}/${boxes.length}`;
  });
}

function applyColumnsDialog() {
  const boxes = $$("#columns-categories input[type='checkbox']");
  const chosen = boxes.filter(b => b.checked).map(b => b.dataset.path);
  state.visiblePaths = chosen.length ? chosen : [...state.defaultColumns];
  saveVisibleColumns(state.parquetName, state.visiblePaths);
  $("#columns-modal").close();
  reloadRows();
}

function setColumnsToDefaults() {
  // Add audio_player as a default since the user likely wants to listen.
  const defaults = [...state.defaultColumns];
  if (!defaults.includes("_audio_player")) defaults.push("_audio_player");
  $$("#columns-categories input[type='checkbox']").forEach(b => {
    b.checked = defaults.includes(b.dataset.path);
  });
}

function setColumnsToAll() {
  $$("#columns-categories input[type='checkbox']").forEach(b => b.checked = true);
}

function setColumnsToNone() {
  $$("#columns-categories input[type='checkbox']").forEach(b => b.checked = false);
}

// ---------- parquet picker ----------
async function populateParquetPicker(active) {
  const data = await loadParquetList();
  const sel = $("#parquet-picker");
  sel.innerHTML = data.parquets.map(p =>
    `<option value="${escapeHtml(p.name)}" ${p.path === active ? "selected" : ""}>${escapeHtml(p.dir)}/${escapeHtml(p.name)} (${p.size_mb} MB)</option>`
  ).join("");
}

async function switchParquet(name) {
  state.parquetName = name;
  state.page = 1;
  // Reload schema + reset visible columns to user's choice for this parquet
  // (or defaults).
  const schemaResp = await loadSchema(name);
  state.schema = schemaResp.columns;
  state.defaultColumns = schemaResp.default_columns;
  const flat = flattenSchema(state.schema);
  state.paths = flat.paths;
  state.pathMeta = flat.meta;
  // Filter user's saved choice to paths still present in the new parquet.
  // Audio player is ALWAYS in the default fallback — listening to clips is
  // the most-common curator action, so the inline player should be visible
  // on first load. User can still hide it via Columns dialog.
  const validDefault = state.defaultColumns.filter(p => state.paths.includes(p));
  const fallback = validDefault.length
    ? [...validDefault, "_audio_player"]
    : [...state.paths.slice(0, 4), "_audio_player"];
  const saved = loadVisibleColumns(state.parquetName, fallback);
  state.visiblePaths = saved.filter(p =>
    p === "_audio_player" || state.paths.includes(p)
  );
  if (!state.visiblePaths.length) state.visiblePaths = fallback;
  state.colWidths = loadColumnWidths();
  // Reload sources + stats + rows
  await refreshSources();
  await refreshStats();
  await reloadRows();
}

async function refreshSources() {
  try {
    const data = await loadSources(state.parquetName);
    const sel = $("#f-source");
    sel.innerHTML = data.map(s =>
      `<option value="${escapeHtml(s.source)}">${escapeHtml(s.source)} (${fmtInt(s.n)})</option>`
    ).join("");
  } catch {}
}

async function refreshStats() {
  const s = await loadStats(state.parquetName);
  renderHeaderMeta(s);
}

async function reloadRows() {
  // Defensive: persist the current column selection on every load. Catches
  // cases where the user closes the dialog or refreshes before Apply.
  if (state.parquetName) {
    saveVisibleColumns(state.parquetName, state.visiblePaths);
  }
  renderTableHeader();
  try {
    const data = await loadRows();
    state.totalRows = data.total;
    renderResultMeta(data);
    renderTableRows(data);
    renderPagination(data);
  } catch (e) {
    $("#rows-body").innerHTML = `<tr><td colspan="${state.visiblePaths.length}" class="empty">load error: ${escapeHtml(e.message)}</td></tr>`;
  }
}

// ---------- event wiring ----------
function wireEvents() {
  $("#parquet-picker").addEventListener("change", (ev) => {
    switchParquet(ev.target.value);
  });

  $("#btn-apply").addEventListener("click", () => {
    state.filters.source = Array.from($("#f-source").selectedOptions).map(o => o.value);
    state.filters.q = ($("#f-search").value || "").trim();
    state.page = 1;
    reloadRows();
  });

  $("#btn-clear").addEventListener("click", () => {
    $("#f-source").selectedIndex = -1;
    $("#f-search").value = "";
    state.filters = { source: [], q: "", ranges: {} };
    state.page = 1;
    reloadRows();
  });

  $("#btn-columns").addEventListener("click", openColumnsDialog);
  $("#columns-close").addEventListener("click", () => $("#columns-modal").close());
  $("#columns-cancel").addEventListener("click", () => $("#columns-modal").close());
  $("#columns-apply").addEventListener("click", applyColumnsDialog);
  $("#columns-defaults").addEventListener("click", setColumnsToDefaults);
  $("#columns-all").addEventListener("click", setColumnsToAll);
  $("#columns-none").addEventListener("click", setColumnsToNone);

  $("#rows-head").addEventListener("click", (ev) => {
    const th = ev.target.closest("th[data-sort]");
    if (!th) return;
    const path = th.dataset.sort;
    if (state.sort === path) {
      state.dir = state.dir === "asc" ? "desc" : "asc";
    } else {
      state.sort = path;
      state.dir = "asc";
    }
    state.page = 1;
    reloadRows();
  });

  $("#pagination").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-page]");
    if (!btn || btn.disabled) return;
    const page = btn.dataset.page;
    if (page === "prev") state.page = Math.max(1, state.page - 1);
    else if (page === "next") {
      const pages = Math.ceil(state.totalRows / state.page_size);
      state.page = Math.min(pages, state.page + 1);
    } else state.page = parseInt(page);
    reloadRows();
  });

  $("#rows-body").addEventListener("click", async (ev) => {
    const a = ev.target.closest(".utt-link");
    if (!a) return;
    ev.preventDefault();
    const uid = a.dataset.utt;
    try {
      const url = `/api/row/${encodeURIComponent(uid)}?parquet=${encodeURIComponent(state.parquetName)}`;
      const data = await api(url);
      $("#detail-utt").textContent = uid;
      $("#detail-json").textContent = JSON.stringify(data, null, 2);
      $("#detail-modal").showModal();
    } catch (e) {
      console.error(e);
    }
  });
  $("#modal-close").addEventListener("click", () => $("#detail-modal").close());

  $("#f-search").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") $("#btn-apply").click();
  });
}

// ---------- init ----------
async function init() {
  wireEvents();
  // Get the active parquet name from the server.
  const list = await loadParquetList();
  const activeName = list.active
    ? list.parquets.find(p => p.path === list.active)?.name
    : (list.parquets[0]?.name);
  if (!activeName) {
    $("#header-meta").textContent = "no parquets found in processed/parquets/ or processed/manifests/";
    return;
  }
  await populateParquetPicker(list.active);
  await switchParquet(activeName);
}

document.addEventListener("DOMContentLoaded", init);
