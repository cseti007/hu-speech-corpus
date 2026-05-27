// Curator frontend — vanilla JS, no framework. Renders the manifest table,
// drives filters, syncs state to the URL hash so views are shareable, and
// pops a detail modal on utterance_id click.
//
// Table layout is fully data-driven (see COLUMNS below) so the user can pick
// which columns to show via the Columns dialog. Choice persists in localStorage.

(() => {
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// === Helpers ===

const fmtInt = (n) => (n ?? 0).toLocaleString("en-US");
const fmtSec = (s) => (s == null ? "—" : `${s.toFixed(1)}s`);
const fmtScore = (s, digits = 2) => (s == null ? "—" : Number(s).toFixed(digits));

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
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

// === Column metadata ===
//
// All possible columns. `default: true` ones are visible out of the box;
// users toggle visibility via the Columns dialog. `poc_only: true` columns
// only appear when the loaded parquet has the qc_* schema (see /api/stats's
// has_qc_columns). `synthetic: true` means the column's value isn't a field
// on the row — the renderer constructs it from other fields (e.g. audio).

const COLUMNS = [
  // Identification
  { id: "utterance_id", label: "utterance_id", category: "Identification",
    default: true, sortable: true, render: "utt_link" },
  { id: "source", label: "source", category: "Identification",
    default: true, sortable: true, render: "source_badge" },

  // Audio
  { id: "duration_sec", label: "dur", category: "Audio",
    default: true, sortable: true, render: "duration", align: "right" },
  { id: "audio", label: "audio", category: "Audio",
    default: true, sortable: false, render: "audio", synthetic: true },

  // Transcripts
  { id: "has_text", label: "has text", category: "Transcripts",
    default: false, sortable: true, render: "bool" },
  { id: "text_preview", label: "text preview", category: "Transcripts",
    default: true, sortable: false, render: "text_preview" },
  { id: "text_qwen_ft_greedy", label: "Qwen text", category: "Transcripts",
    default: false, sortable: false, render: "text_long", poc_only: true },
  { id: "text_canary_v2_greedy", label: "Canary text", category: "Transcripts",
    default: false, sortable: false, render: "text_long", poc_only: true },

  // Quality flags
  { id: "qf_dnsmos_ovrl", label: "DNS", category: "Quality flags",
    default: true, sortable: true, render: "score_dns", align: "right" },
  { id: "qf_vad_speech_ratio", label: "VAD", category: "Quality flags",
    default: true, sortable: true, render: "score_vad", align: "right" },
  { id: "qf_lid", label: "LID", category: "Quality flags",
    default: true, sortable: true, render: "lid" },
  { id: "qf_lid_is_hu_prob", label: "HU prob", category: "Quality flags",
    default: false, sortable: true, render: "score", align: "right" },
  { id: "qf_is_clipped", label: "clip", category: "Quality flags",
    default: false, sortable: true, render: "bool" },

  // PoC consensus metrics (only on manifest_poc_100h.parquet)
  { id: "qc_pairwise_wer", label: "Q↔C WER", category: "PoC metrics",
    default: false, sortable: true, render: "score_wer", align: "right",
    poc_only: true },
  { id: "qc_exact_match", label: "GOLD", category: "PoC metrics",
    default: false, sortable: true, render: "gold_badge", poc_only: true },
];

const COLUMN_BY_ID = Object.fromEntries(COLUMNS.map((c) => [c.id, c]));
const LS_VISIBLE_COLS = "curator.visibleColumns.v1";

const CELL_RENDERERS = {
  utt_link: (v) =>
    `<a class="utt-link" data-utt="${escapeHtml(v)}" href="#">${escapeHtml(v)}</a>`,
  source_badge: (v) =>
    `<span class="src-badge src-${escapeHtml(v)}">${escapeHtml(v)}</span>`,
  duration: (v) => fmtSec(v),
  text_preview: (v, r) => r.has_text
    ? `<div class="text-preview has-text" title="${escapeHtml(v)}">${escapeHtml(v)}</div>`
    : `<div class="text-preview text-no">(audio only)</div>`,
  text_long: (v) => v
    ? `<div class="text-preview has-text" title="${escapeHtml(v)}">${escapeHtml(truncate(v, 80))}</div>`
    : `<div class="text-preview text-no">—</div>`,
  score_dns: (v) =>
    `<span class="${scoreClass(v, 3.0, 2.0)}">${fmtScore(v)}</span>`,
  score_vad: (v) =>
    `<span class="${scoreClass(v, 0.7, 0.3)}">${fmtScore(v)}</span>`,
  lid: (v) => {
    const cls = v == null ? "score-na" : (v === "hu" ? "score-good" : "score-bad");
    return `<span class="${cls}">${escapeHtml(v ?? "—")}</span>`;
  },
  score: (v) => fmtScore(v),
  bool: (v) => {
    if (v === true)  return '<span class="bool-yes">✓</span>';
    if (v === false) return '<span class="bool-no">×</span>';
    return '<span class="score-na">—</span>';
  },
  audio: (_, r) => {
    const orig = r.relative_audio_path;
    const refined = r.refined_relative_audio_path;
    if (!orig && !refined) return '<span class="audio-na">no audio</span>';
    const parts = [];
    if (orig) {
      parts.push(
        '<div class="audio-row"><span class="audio-label">orig</span>' +
        `<audio controls preload="none" src="/audio/${encodeURI(orig)}"></audio></div>`
      );
    }
    if (refined) {
      parts.push(
        '<div class="audio-row"><span class="audio-label">refined</span>' +
        `<audio controls preload="none" src="/audio/${encodeURI(refined)}"></audio></div>`
      );
    }
    return parts.join("");
  },
  score_wer: (v) => {
    if (v == null) return '<span class="score-na">—</span>';
    const cls = v <= 0.05 ? "score-good" : v >= 0.5 ? "score-bad" : "score-warn";
    return `<span class="${cls}">${(v * 100).toFixed(1)}%</span>`;
  },
  gold_badge: (v) => v === true
    ? '<span class="gold-badge">GOLD</span>'
    : '<span class="score-na">—</span>',
};

// === State ===

const state = {
  filters: {},
  sort: "utterance_id",
  dir: "asc",
  page: 1,
  page_size: 50,
  visibleColumnIds: new Set(),   // populated at init
  hasQcColumns: false,
};

// === Visibility persistence ===

function defaultVisibleIds(hasQc) {
  return COLUMNS
    .filter((c) => c.default && (!c.poc_only || hasQc))
    .map((c) => c.id);
}

function loadVisibleColumns(hasQc) {
  try {
    const raw = localStorage.getItem(LS_VISIBLE_COLS);
    if (raw) {
      const arr = JSON.parse(raw);
      if (Array.isArray(arr)) {
        // Drop any saved ids that no longer correspond to known columns.
        const known = arr.filter((id) => id in COLUMN_BY_ID);
        if (known.length) return new Set(known);
      }
    }
  } catch (e) { /* fall through to defaults */ }
  return new Set(defaultVisibleIds(hasQc));
}

function saveVisibleColumns(idSet) {
  try {
    localStorage.setItem(LS_VISIBLE_COLS, JSON.stringify(Array.from(idSet)));
  } catch (e) { /* ignore quota errors */ }
}

function visibleColumns() {
  // Intersection of: user-chosen ids × columns applicable to this parquet.
  return COLUMNS.filter((c) =>
    state.visibleColumnIds.has(c.id) && (!c.poc_only || state.hasQcColumns)
  );
}

// === URL <-> state ===

function stateToHash() {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) {
      for (const item of v) params.append(k, item);
    } else {
      params.set(k, v);
    }
  }
  if (state.sort !== "utterance_id") params.set("sort", state.sort);
  if (state.dir !== "asc") params.set("dir", state.dir);
  if (state.page !== 1) params.set("page", String(state.page));
  if (state.page_size !== 50) params.set("page_size", String(state.page_size));
  const s = params.toString();
  history.replaceState(null, "", s ? `#${s}` : "#");
}

function hashToState() {
  const hash = window.location.hash.replace(/^#/, "");
  const params = new URLSearchParams(hash);
  const filters = {};
  for (const key of params.keys()) {
    if (["sort", "dir", "page", "page_size"].includes(key)) continue;
    const all = params.getAll(key);
    filters[key] = all.length > 1 ? all : all[0];
  }
  state.filters = filters;
  state.sort = params.get("sort") || "utterance_id";
  state.dir = params.get("dir") || "asc";
  state.page = parseInt(params.get("page") || "1", 10);
  state.page_size = parseInt(params.get("page_size") || "50", 10);
}

// === Form <-> state ===

function readForm() {
  const f = {};
  const sources = Array.from($("#f-source").selectedOptions)
    .map((o) => o.value).filter(Boolean);
  if (sources.length) f.source = sources;
  const setIf = (key, val) => { if (val !== "" && val != null) f[key] = val; };
  setIf("has_text", $("#f-has-text").value);
  setIf("lid", $("#f-lid").value);
  setIf("is_clipped", $("#f-clipped").value);
  setIf("halluc", $("#f-halluc").value);
  setIf("foreign", $("#f-foreign")?.value);
  setIf("foreign_prefix_min", $("#f-fp-min")?.value);
  setIf("duration_min", $("#f-dur-min").value);
  setIf("duration_max", $("#f-dur-max").value);
  setIf("dnsmos_min", $("#f-dnsmos-min").value);
  setIf("dnsmos_max", $("#f-dnsmos-max").value);
  setIf("vad_min", $("#f-vad-min").value);
  setIf("vad_max", $("#f-vad-max").value);
  setIf("lid_hu_min", $("#f-lid-hu-min").value);
  setIf("lid_hu_max", $("#f-lid-hu-max").value);
  // PoC-only widgets (no-op when the form row is hidden)
  setIf("qc_exact", $("#f-qc-exact")?.value);
  setIf("qc_wer_min", $("#f-qc-wer-min")?.value);
  setIf("qc_wer_max", $("#f-qc-wer-max")?.value);
  const q = $("#f-search").value.trim();
  if (q) f.q = q;
  state.filters = f;
}

function applyFiltersToForm() {
  const f = state.filters;
  const setVal = (id, val) => { const el = $(id); if (el) el.value = val ?? ""; };
  setVal("#f-has-text", f.has_text);
  setVal("#f-lid", f.lid);
  setVal("#f-clipped", f.is_clipped);
  setVal("#f-halluc", f.halluc);
  setVal("#f-foreign", f.foreign);
  setVal("#f-fp-min", f.foreign_prefix_min);
  setVal("#f-dur-min", f.duration_min);
  setVal("#f-dur-max", f.duration_max);
  setVal("#f-dnsmos-min", f.dnsmos_min);
  setVal("#f-dnsmos-max", f.dnsmos_max);
  setVal("#f-vad-min", f.vad_min);
  setVal("#f-vad-max", f.vad_max);
  setVal("#f-lid-hu-min", f.lid_hu_min);
  setVal("#f-lid-hu-max", f.lid_hu_max);
  setVal("#f-qc-exact", f.qc_exact);
  setVal("#f-qc-wer-min", f.qc_wer_min);
  setVal("#f-qc-wer-max", f.qc_wer_max);
  setVal("#f-search", f.q);
  const sourceSel = $("#f-source");
  const want = new Set(Array.isArray(f.source)
    ? f.source : (f.source ? [f.source] : []));
  for (const opt of sourceSel.options) opt.selected = want.has(opt.value);
}

// === Fetch ===

function buildQuery() {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) for (const item of v) params.append(k, item);
    else params.set(k, v);
  }
  params.set("sort", state.sort);
  params.set("dir", state.dir);
  params.set("page", String(state.page));
  params.set("page_size", String(state.page_size));
  return params.toString();
}

async function fetchRows() {
  const cols = visibleColumns();
  const tbody = $("#rows-body");
  tbody.innerHTML =
    `<tr><td colspan="${cols.length || 1}" class="msg-loading">loading…</td></tr>`;
  renderHeader(cols);
  const url = `/api/rows?${buildQuery()}`;
  const res = await fetch(url);
  if (!res.ok) {
    tbody.innerHTML =
      `<tr><td colspan="${cols.length || 1}" class="msg-error">error: ${res.status}</td></tr>`;
    return;
  }
  const data = await res.json();
  renderRows(data, cols);
  renderPagination(data);
  renderResultMeta(data);
  updateSortHeaders();
  stateToHash();
}

// === Render ===

function renderResultMeta(data) {
  const from = data.total === 0 ? 0 : (data.page - 1) * data.page_size + 1;
  const to = Math.min(data.total, data.page * data.page_size);
  $("#result-meta").textContent =
    `Showing ${fmtInt(from)}–${fmtInt(to)} of ${fmtInt(data.total)} matches`;
}

function renderHeader(cols) {
  const ths = cols.map((c) => {
    const align = c.align === "right" ? " num" : "";
    const sortable = c.sortable ? " sortable" : "";
    const dataAttr = c.sortable ? ` data-col="${c.id}"` : "";
    return `<th class="col-${c.id}${align}${sortable}"${dataAttr}>${escapeHtml(c.label)}</th>`;
  });
  $("#rows-head").innerHTML = `<tr>${ths.join("")}</tr>`;
}

function renderRows(data, cols) {
  const tbody = $("#rows-body");
  if (!data.rows.length) {
    tbody.innerHTML =
      `<tr><td colspan="${cols.length}" class="msg-empty">no rows match the filters</td></tr>`;
    return;
  }
  const html = data.rows.map((r) => renderRow(r, cols)).join("");
  tbody.innerHTML = html;
}

function renderRow(r, cols) {
  const cells = cols.map((c) => {
    const value = c.synthetic ? null : r[c.id];
    const renderer = CELL_RENDERERS[c.render];
    const inner = renderer ? renderer(value, r) : escapeHtml(value ?? "—");
    const align = c.align === "right" ? " num" : "";
    return `<td class="col-${c.id}${align}">${inner}</td>`;
  });
  return `<tr>${cells.join("")}</tr>`;
}

function renderPagination(data) {
  const total_pages = Math.max(1, Math.ceil(data.total / data.page_size));
  const cur = data.page;
  const el = $("#pagination");
  if (data.total === 0) { el.innerHTML = ""; return; }

  const btn = (label, page, cls = "") =>
    `<button class="pg-btn ${cls}" data-page="${page}">${label}</button>`;
  const ellipsis = () => '<span class="pg-ellipsis">…</span>';

  const parts = [];
  parts.push(btn("« prev", cur - 1, cur <= 1 ? "disabled" : ""));
  const pages = new Set([1, total_pages, cur - 2, cur - 1, cur, cur + 1, cur + 2]);
  const visible = Array.from(pages)
    .filter((p) => p >= 1 && p <= total_pages).sort((a, b) => a - b);
  let prev = 0;
  for (const p of visible) {
    if (p - prev > 1) parts.push(ellipsis());
    parts.push(btn(String(p), p, p === cur ? "active" : ""));
    prev = p;
  }
  parts.push(btn("next »", cur + 1, cur >= total_pages ? "disabled" : ""));

  el.innerHTML = parts.join("");
  for (const b of el.querySelectorAll(".pg-btn")) {
    b.addEventListener("click", () => {
      const p = parseInt(b.dataset.page, 10);
      if (p < 1 || p > total_pages) return;
      state.page = p;
      fetchRows();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
}

function updateSortHeaders() {
  for (const th of $$("#rows-head th.sortable")) {
    th.classList.remove("sort-asc", "sort-desc");
    if (th.dataset.col === state.sort) {
      th.classList.add(state.dir === "asc" ? "sort-asc" : "sort-desc");
    }
  }
}

// === Detail modal ===

async function openDetail(utt) {
  const modal = $("#detail-modal");
  $("#detail-utt").textContent = utt;
  $("#detail-json").textContent = "loading…";
  modal.showModal();
  try {
    const res = await fetch(`/api/row/${encodeURIComponent(utt)}`);
    if (!res.ok) {
      $("#detail-json").textContent = `error: ${res.status}`;
      return;
    }
    const data = await res.json();
    for (const k of Object.keys(data)) {
      if (k.endsWith("_json") && typeof data[k] === "string" && data[k]) {
        try { data[k] = JSON.parse(data[k]); } catch (e) { /* keep as-is */ }
      }
    }
    $("#detail-json").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("#detail-json").textContent = `error: ${e.message}`;
  }
}

// === Column picker dialog ===

function populateColumnsDialog() {
  const container = $("#columns-categories");
  const categories = [];
  const seenCats = new Set();
  for (const c of COLUMNS) {
    if (c.poc_only && !state.hasQcColumns) continue;
    if (!seenCats.has(c.category)) {
      categories.push(c.category);
      seenCats.add(c.category);
    }
  }
  const html = categories.map((cat) => {
    const cols = COLUMNS.filter((c) =>
      c.category === cat && (!c.poc_only || state.hasQcColumns)
    );
    const rows = cols.map((c) => {
      const checked = state.visibleColumnIds.has(c.id) ? "checked" : "";
      const pocTag = c.poc_only ? ' <span class="poc-tag">PoC</span>' : "";
      return `<label class="col-pick-row">
        <input type="checkbox" data-col-id="${c.id}" ${checked}>
        <span class="col-pick-label">${escapeHtml(c.label)}${pocTag}</span>
        <span class="col-pick-id">${escapeHtml(c.id)}</span>
      </label>`;
    }).join("");
    return `<div class="col-pick-cat">
      <h3>${escapeHtml(cat)}</h3>
      ${rows}
    </div>`;
  }).join("");
  container.innerHTML = html;
}

function openColumnsDialog() {
  populateColumnsDialog();
  $("#columns-modal").showModal();
}

function readColumnsDialog() {
  const ids = new Set();
  for (const cb of $$("#columns-categories input[type=checkbox]")) {
    if (cb.checked) ids.add(cb.dataset.colId);
  }
  return ids;
}

function setDialogCheckboxes(predicate) {
  for (const cb of $$("#columns-categories input[type=checkbox]")) {
    cb.checked = predicate(cb.dataset.colId);
  }
}

// === Init ===

async function init() {
  // Stats first — drives parquet capability detection
  let hasQcColumns = false;
  try {
    const s = await fetch("/api/stats").then((r) => r.json());
    const parquet = s.parquet_name ? ` · loaded: ${s.parquet_name}` : "";
    $("#header-meta").textContent =
      `${fmtInt(s.total_rows)} rows · ${s.total_hours.toLocaleString()} h${parquet}`;
    hasQcColumns = !!s.has_qc_columns;
  } catch (e) {
    $("#header-meta").textContent = "error loading stats";
  }
  state.hasQcColumns = hasQcColumns;
  state.visibleColumnIds = loadVisibleColumns(hasQcColumns);

  if (hasQcColumns) {
    const row = $("#filter-poc-row");
    if (row) row.style.display = "";
  }

  // Sources dropdown
  try {
    const sources = await fetch("/api/sources").then((r) => r.json());
    const sel = $("#f-source");
    sel.innerHTML = sources.map((s) =>
      `<option value="${escapeHtml(s.source)}">${escapeHtml(s.source)} (${fmtInt(s.n)})</option>`
    ).join("");
  } catch (e) { /* leave empty */ }

  // URL → state → form
  hashToState();
  applyFiltersToForm();

  // Filter bar events
  $("#btn-apply").addEventListener("click", () => {
    readForm();
    state.page = 1;
    fetchRows();
  });
  $("#btn-clear").addEventListener("click", () => {
    for (const el of $$("input.search, .filter input, .filter select")) {
      if (el.tagName === "SELECT" && el.multiple) {
        for (const opt of el.options) opt.selected = false;
      } else {
        el.value = "";
      }
    }
    state.filters = {};
    state.page = 1;
    fetchRows();
  });
  $("#f-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { readForm(); state.page = 1; fetchRows(); }
  });

  // Sort header click (event delegation — survives dynamic re-renders)
  $("#rows-head").addEventListener("click", (ev) => {
    const th = ev.target.closest("th.sortable");
    if (!th) return;
    const col = th.dataset.col;
    if (state.sort === col) state.dir = state.dir === "asc" ? "desc" : "asc";
    else { state.sort = col; state.dir = "asc"; }
    state.page = 1;
    fetchRows();
  });

  // utt-link click (event delegation — survives dynamic re-renders)
  $("#rows-body").addEventListener("click", (ev) => {
    const link = ev.target.closest(".utt-link");
    if (!link) return;
    ev.preventDefault();
    openDetail(link.dataset.utt);
  });

  // Detail modal close
  $("#modal-close").addEventListener("click", () => $("#detail-modal").close());
  $("#detail-modal").addEventListener("click", (e) => {
    if (e.target.id === "detail-modal") $("#detail-modal").close();
  });

  // Columns dialog
  $("#btn-columns").addEventListener("click", openColumnsDialog);
  $("#columns-close").addEventListener("click", () => $("#columns-modal").close());
  $("#columns-cancel").addEventListener("click", () => $("#columns-modal").close());
  $("#columns-apply").addEventListener("click", () => {
    state.visibleColumnIds = readColumnsDialog();
    saveVisibleColumns(state.visibleColumnIds);
    $("#columns-modal").close();
    fetchRows();  // re-render header + body
  });
  $("#columns-defaults").addEventListener("click", () => {
    const defaults = new Set(defaultVisibleIds(state.hasQcColumns));
    setDialogCheckboxes((id) => defaults.has(id));
  });
  $("#columns-all").addEventListener("click", () => {
    setDialogCheckboxes(() => true);
  });
  $("#columns-none").addEventListener("click", () => {
    setDialogCheckboxes(() => false);
  });
  $("#columns-modal").addEventListener("click", (e) => {
    if (e.target.id === "columns-modal") $("#columns-modal").close();
  });

  // Esc closes whichever modal is open
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if ($("#detail-modal").open) $("#detail-modal").close();
    if ($("#columns-modal").open) $("#columns-modal").close();
  });

  // URL hash change (e.g. user pastes a new hash without full reload)
  window.addEventListener("hashchange", () => {
    hashToState();
    applyFiltersToForm();
    fetchRows();
  });

  // Initial render
  fetchRows();
}

init();

})();
