// ══════════════════════════════════════════════════════════
//  edu_pipeline app.js  — Full Feature Build
// ══════════════════════════════════════════════════════════

// ── SVG 아이콘 라이브러리 ───────────────────────────────────────────────
const ICONS = {
  book:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  memo:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg>`,
  bulb:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>`,
  key:        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="3.5"/><path d="M21 2l-9.6 9.6M15.5 7.5l2 2M17 6l2 2"/></svg>`,
  question:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  doc:        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
  play:       `<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`,
  clip:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>`,
  send:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2" fill="currentColor" stroke="none"/></svg>`,
  moon:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`,
  sun:        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
  trash:      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
  close:      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  plus:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  check:      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  'check-ok': `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  copy:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  chat:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  search:     `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  film:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>`,
  ai:         `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8 8V5.5a4 4 0 0 1 8 0V8"/><circle cx="9" cy="14" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="14" r="1" fill="currentColor" stroke="none"/><line x1="9.5" y1="17.5" x2="14.5" y2="17.5"/></svg>`,
  user:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.58-7 8-7s8 3 8 7"/></svg>`,
  download:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  'chev-d':   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`,
  'chev-u':   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`,
};

/** 아이콘 이름 → HTML 문자열 */
function ic(name) {
  return `<span class="ic">${ICONS[name] || ''}</span>`;
}

// ── State ──────────────────────────────────────────────────────────
let activeJobId     = null;
let selectedCols    = new Set();
let allCols         = [];
let isProcessing    = false;
let isChatting      = false;
let lastQuestion    = null;   // 재생성용
let ragThreshold    = 0.6;
let targetLang      = 'ko';
let activeVideoStem = null;

// 히스토리
const LS_KEY       = 'edu_sessions';
const LS_COLS_KEY  = 'edu_selected_cols';   // 선택된 컬렉션 유지
let sessions       = [];
let currentSession = null;

// 소스 정보 캐시
const sourceInfoCache = {};

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyDarkMode();
  loadHistory();
  restoreSelectedCols();
  loadCollections();
  startNewSession();
  requestNotificationPermission();
  pollReadyState();
});

// ── AI 초기화 상태 폴링 ────────────────────────────────────
async function pollReadyState() {
  const banner = document.getElementById('ai-loading-banner');
  const msgEl  = document.getElementById('alb-msg');
  if (!banner) return;

  // 먼저 한 번 체크해서 이미 ready면 바로 숨김
  try {
    const res  = await fetch('/api/ready');
    const data = await res.json();
    if (data.ready) { banner.style.display = 'none'; return; }
    if (msgEl) msgEl.textContent = data.message || 'AI 로딩 중...';
  } catch { banner.style.display = 'none'; return; }

  banner.style.display = 'flex';

  const iv = setInterval(async () => {
    try {
      const res  = await fetch('/api/ready');
      const data = await res.json();
      if (msgEl) msgEl.textContent = data.message || 'AI 로딩 중...';
      if (data.ready) {
        clearInterval(iv);
        banner.classList.add('alb-done');
        setTimeout(() => { banner.style.display = 'none'; }, 800);
      }
    } catch {
      clearInterval(iv);
      banner.style.display = 'none';
    }
  }, 1500);
}

// 페이지 이탈 전 현재 대화 자동 저장
window.addEventListener('beforeunload', () => {
  saveSession();
  saveSelectedCols();
});

// ══════════════════════════════════════════════════════════
//  다크 모드
// ══════════════════════════════════════════════════════════
function applyDarkMode() {
  const dark = localStorage.getItem('dark') === '1';
  document.body.classList.toggle('dark', dark);
  const btn = document.getElementById('dark-btn');
  if (btn) btn.innerHTML = ic(dark ? 'sun' : 'moon');
}

function toggleDark() {
  const isDark = document.body.classList.toggle('dark');
  localStorage.setItem('dark', isDark ? '1' : '0');
  const btn = document.getElementById('dark-btn');
  if (btn) btn.innerHTML = ic(isDark ? 'sun' : 'moon');
}

// ══════════════════════════════════════════════════════════
//  모바일 사이드바
// ══════════════════════════════════════════════════════════
function toggleSidebar() {
  document.querySelector('.sidebar')?.classList.toggle('open');
  document.getElementById('sidebar-overlay')?.classList.toggle('open');
}
function closeSidebar() {
  document.querySelector('.sidebar')?.classList.remove('open');
  document.getElementById('sidebar-overlay')?.classList.remove('open');
}

// ══════════════════════════════════════════════════════════
//  알림 권한
// ══════════════════════════════════════════════════════════
function requestNotificationPermission() {
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }
}

function sendNotification(title, body) {
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(title, { body, icon: '/static/icon.png' });
  }
}

// ══════════════════════════════════════════════════════════
//  컬렉션 목록
// ══════════════════════════════════════════════════════════
async function loadCollections() {
  try {
    const res = await fetch('/api/collections');
    allCols   = await res.json();
    // 이미 선택된 컬렉션 중 더 이상 존재하지 않는 것 제거
    const existing = new Set(allCols);
    for (const c of [...selectedCols]) {
      if (!existing.has(c)) selectedCols.delete(c);
    }
    renderCollections();
    renderSourceBar();
    allCols.forEach(name => prefetchSourceInfo(name));
    // 라이브러리 탭이 열려있으면 함께 갱신
    if (document.getElementById('tab-library')?.style.display !== 'none') {
      renderLibrary();
    }
  } catch {}
}

async function prefetchSourceInfo(stem) {
  if (sourceInfoCache[stem]) return;
  try {
    const [infoRes, segRes] = await Promise.all([
      fetch(`/api/source-info/${encodeURIComponent(stem)}`),
      fetch(`/api/content-preview/${encodeURIComponent(stem)}`),
    ]);
    const info = await infoRes.json();
    const seg  = segRes.ok ? await segRes.json() : {};
    sourceInfoCache[stem] = {
      ...info,
      segment_count:     seg.count    ?? null,
      duration_seconds:  seg.duration ?? null,
    };
    // 사이드바 부제목 갱신
    const el = document.getElementById(`sc-type-${stem}`);
    if (el && info?.title && info.title !== stem) el.textContent = info.title;
    // 라이브러리 탭이 열려있으면 전체 재렌더 (그룹 구조 반영)
    if (document.getElementById('tab-library')?.style.display !== 'none') {
      renderLibrary();
    }
    // 소스바 칩 라벨도 갱신
    renderSourceBar();
  } catch {}
}

function renderCollections() {
  const el = document.getElementById('col-list');
  if (!allCols.length) {
    el.innerHTML = '<div class="empty-hint">아직 처리된 파일이<br>없습니다.<br><br>파일을 업로드하면<br>여기에 표시됩니다.</div>';
    return;
  }
  // 사이드바: 선택 없이 정보 확인 전용 (선택은 채팅창 소스 바에서)
  el.innerHTML = allCols.map(name => {
    const info  = sourceInfoCache[name];
    const isExp = info?.data_type === 'experimental';
    const sub   = (info?.title && info.title !== name) ? info.title
                : isExp ? '실험 데이터' : '영상 컬렉션';
    const badge = isExp ? '<span class="sc-badge-exp">실험</span>' : '';
    return `
    <div class="source-card" onclick="openInfoPanel(event,'${name}')"
         role="button" tabindex="0" aria-label="${name} 소스 정보 보기"
         onkeydown="if(event.key==='Enter')openInfoPanel(event,'${name}')">
      <div class="sc-top">
        <div style="min-width:0;flex:1">
          <div class="sc-name" title="${name}">${name}${badge}</div>
          <div class="sc-type" id="sc-type-${name}">${sub}</div>
        </div>
        <button class="sc-info-btn" onclick="openInfoPanel(event,'${name}')"
                aria-label="${name} 상세 정보">ℹ</button>
      </div>
    </div>`;
  }).join('');
}

function toggleCol(name) {
  if (selectedCols.has(name)) selectedCols.delete(name);
  else selectedCols.add(name);
  saveSelectedCols();
  renderSourceBar();
  // 피커가 열려있으면 해당 항목 체크 상태 갱신
  const item = document.getElementById(`sp-item-${CSS.escape(name)}`);
  if (item) {
    item.classList.toggle('selected', selectedCols.has(name));
    const cb = item.querySelector('input[type=checkbox]');
    if (cb) cb.checked = selectedCols.has(name);
  }
}

function saveSelectedCols() {
  try { localStorage.setItem(LS_COLS_KEY, JSON.stringify([...selectedCols])); } catch {}
}

function restoreSelectedCols() {
  try {
    const saved = JSON.parse(localStorage.getItem(LS_COLS_KEY) || '[]');
    saved.forEach(c => selectedCols.add(c));
  } catch {}
}

// ══════════════════════════════════════════════════════════
//  채팅 입력 소스 바 & 피커
// ══════════════════════════════════════════════════════════

function renderSourceBar() {
  const bar = document.getElementById('source-bar');
  if (!bar) return;
  if (!allCols.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';

  const selArr = [...selectedCols].filter(n => allCols.includes(n));
  let html = '';

  if (selArr.length === 0) {
    html += `<span class="src-all-chip">전체 소스 (${allCols.length}개)</span>`;
  } else {
    html += selArr.map(name => {
      const label = sourceInfoCache[name]?.title || name;
      const safe  = name.replace(/'/g, "\\'");
      return `<span class="src-chip" title="${name}">
        <span class="src-chip-label">${label}</span>
        <button class="src-chip-remove" onclick="toggleCol('${safe}')"
                aria-label="${label} 선택 해제">×</button>
      </span>`;
    }).join('');
  }

  html += `<button class="src-add-btn" id="src-add-btn"
           onclick="toggleSourcePicker()" aria-label="소스 선택" aria-haspopup="true"
           aria-expanded="false">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    소스 선택
  </button>`;
  bar.innerHTML = html;
}

function toggleSourcePicker() {
  const picker = document.getElementById('source-picker');
  if (!picker) return;
  const open = picker.classList.contains('open');
  if (open) {
    picker.classList.remove('open');
    document.getElementById('src-add-btn')?.setAttribute('aria-expanded', 'false');
  } else {
    renderSourcePickerItems();
    picker.classList.add('open');
    document.getElementById('src-add-btn')?.setAttribute('aria-expanded', 'true');
    // 첫 번째 체크박스로 포커스
    picker.querySelector('input[type=checkbox]')?.focus();
  }
}

function renderSourcePickerItems() {
  const list = document.getElementById('sp-list');
  if (!list) return;
  if (!allCols.length) {
    list.innerHTML = '<div class="sp-empty">처리된 소스가 없습니다</div>';
    return;
  }
  list.innerHTML = allCols.map(name => {
    const info  = sourceInfoCache[name];
    const label = info?.title || name;
    const sub   = name !== label ? name : '';
    const sel   = selectedCols.has(name);
    const safe  = name.replace(/'/g, "\\'");
    return `<label class="sp-item ${sel ? 'selected' : ''}" id="sp-item-${CSS.escape(name)}"
                   role="option" aria-selected="${sel}">
      <input type="checkbox" ${sel ? 'checked' : ''}
             onchange="toggleCol('${safe}')"
             aria-label="${label} 선택">
      <span class="sp-item-label">${label}</span>
      ${sub ? `<span class="sp-item-sub">${sub}</span>` : ''}
    </label>`;
  }).join('');
}

function spSelectAll() {
  allCols.forEach(n => selectedCols.add(n));
  saveSelectedCols();
  renderSourcePickerItems();
  renderSourceBar();
}

function spDeselectAll() {
  selectedCols.clear();
  saveSelectedCols();
  renderSourcePickerItems();
  renderSourceBar();
}

// 피커 외부 클릭 시 닫기
document.addEventListener('click', e => {
  const picker = document.getElementById('source-picker');
  const bar    = document.getElementById('source-bar');
  if (!picker?.classList.contains('open')) return;
  if (!picker.contains(e.target) && !bar?.contains(e.target)) {
    picker.classList.remove('open');
    document.getElementById('src-add-btn')?.setAttribute('aria-expanded', 'false');
  }
});
// Escape 키로 피커 닫기
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const picker = document.getElementById('source-picker');
    if (picker?.classList.contains('open')) {
      picker.classList.remove('open');
      document.getElementById('src-add-btn')?.focus();
    }
  }
});

// ── 컬렉션 삭제 ────────────────────────────────────────────
async function deleteCollection(stem) {
  if (!confirm(`"${stem}" 컬렉션을 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.`)) return;
  try {
    const res  = await fetch(`/api/collection/${encodeURIComponent(stem)}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.ok) {
      allCols = allCols.filter(c => c !== stem);
      selectedCols.delete(stem);
      saveSelectedCols();
      delete sourceInfoCache[stem];
      renderCollections();
      closeInfoPanel();
      addSystemMsg(`"${stem}" 컬렉션이 삭제되었습니다.`);
    } else {
      alert('삭제 실패: ' + (data.error || '알 수 없는 오류'));
    }
  } catch (e) { alert('삭제 실패: ' + e.message); }
}

// ══════════════════════════════════════════════════════════
//  라이브러리 뷰
// ══════════════════════════════════════════════════════════

// 라이브러리 선택 상태 (독립적 — 채팅 selectedCols와 별개)
const libSelected = new Set();

function fmtDuration(sec) {
  if (!sec || sec < 0) return null;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}시간 ${m}분`;
  if (m > 0) return `${m}분 ${s}초`;
  return `${s}초`;
}

let _categorizingAll = false;

function renderLibrary() {
  const grid = document.getElementById('lib-grid');
  if (!grid) return;

  if (!allCols.length) {
    grid.innerHTML = `
      <div class="lib-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
        <div>처리된 소스가 없습니다</div>
        <div>소스 추가 탭에서 영상이나 문서를 업로드하면 여기에 표시됩니다</div>
      </div>`;
    updateLibSelBar();
    return;
  }

  // 태그 집계 — 각 태그가 몇 개 소스에 달려있는지
  const UNCATEGORIZED = '미분류';
  const tagCount = {};
  const sourceTags = {};  // { name: [tags...] }
  for (const name of allCols) {
    const info = sourceInfoCache[name] || {};
    let tags = info.topics;
    if (!tags || !tags.length) tags = info.topic ? [info.topic] : [UNCATEGORIZED];
    sourceTags[name] = tags;
    for (const t of tags) tagCount[t] = (tagCount[t] || 0) + 1;
  }

  const sortedTags = Object.keys(tagCount).sort((a, b) => {
    if (a === UNCATEGORIZED) return 1;
    if (b === UNCATEGORIZED) return -1;
    return a.localeCompare(b, 'ko');
  });

  const hasUncategorized = !!tagCount[UNCATEGORIZED];

  // 활성 필터 (기본: 전체)
  if (typeof _libActiveTag === 'undefined') _libActiveTag = '__all__';
  if (_libActiveTag !== '__all__' && !tagCount[_libActiveTag]) _libActiveTag = '__all__';

  // 필터링된 소스 (중복 없이 flat)
  const visible = _libActiveTag === '__all__'
    ? allCols
    : allCols.filter(n => (sourceTags[n] || []).includes(_libActiveTag));

  // 태그 칩 바
  let html = `<div class="lib-tag-bar" style="display:flex;flex-wrap:wrap;gap:6px;padding:8px 4px 16px;border-bottom:1px solid #222;margin-bottom:16px">
    <span class="lib-tag-chip ${_libActiveTag === '__all__' ? 'active' : ''}"
          onclick="setLibTag('__all__')"
          style="padding:4px 12px;border-radius:14px;cursor:pointer;font-size:12px;
                 background:${_libActiveTag === '__all__' ? '#4a8cff' : '#222'};
                 color:${_libActiveTag === '__all__' ? '#fff' : '#aaa'}">전체 (${allCols.length})</span>`;
  for (const t of sortedTags) {
    const active = _libActiveTag === t;
    html += `<span class="lib-tag-chip ${active ? 'active' : ''}"
          onclick="setLibTag(${JSON.stringify(t)})"
          style="padding:4px 12px;border-radius:14px;cursor:pointer;font-size:12px;
                 background:${active ? '#4a8cff' : '#222'};
                 color:${active ? '#fff' : '#aaa'}">${t} (${tagCount[t]})</span>`;
  }
  html += `</div>`;

  // 카드 그리드 (flat)
  html += `<div class="lib-group-grid">${visible.map(n => renderLibCard(n)).join('')}</div>`;

  // 분류 작업바
  let statusText;
  if (_categorizingAll && _catProgress) {
    const p = _catProgress;
    statusText = `분류 중 ${p.done}/${p.total} (${p.percent || 0}%)` +
                 (p.current ? ` — ${p.current.slice(0, 30)}` : '');
  } else {
    statusText = hasUncategorized
      ? (groups[UNCATEGORIZED].length + '개 소스 미분류')
      : '모든 소스 분류 완료';
  }
  const barWidth = (_categorizingAll && _catProgress) ? (_catProgress.percent || 0) : 0;
  html += `
    <div class="lib-categorize-bar" id="lib-categorize-bar" style="flex-direction:column;align-items:stretch;gap:6px">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
        <span style="font-size:12px">${statusText}</span>
        <span style="display:flex;gap:8px">
          ${hasUncategorized ? `<button id="lib-cat-btn" onclick="categorizeAll()" ${_categorizingAll ? 'disabled' : ''}>
            ${_categorizingAll ? '<span class="lib-cat-spinner"></span> 분류 중...' : ic('ai') + ' 미분류 분류'}
          </button>` : ''}
          <button onclick="reclassifyAll()" ${_categorizingAll ? 'disabled' : ''}>
            ${ic('ai')} 전체 재분류
          </button>
        </span>
      </div>
      ${_categorizingAll ? `<div style="height:6px;background:#222;border-radius:3px;overflow:hidden">
        <div style="height:100%;width:${barWidth}%;background:linear-gradient(90deg,#4a8cff,#7d5fff);transition:width .3s"></div>
      </div>` : ''}
    </div>`;

  grid.innerHTML = html;
  updateLibSelBar();
}

let _libActiveTag = '__all__';
function setLibTag(t) { _libActiveTag = t; renderLibrary(); }

let _catProgress = null;   // {percent, done, total, current}

async function reclassifyAll() {
  if (_categorizingAll) return;
  if (!confirm('모든 소스를 다시 분류할까요? (시간이 좀 걸립니다)')) return;
  _categorizingAll = true;
  _catProgress = { percent: 0, done: 0, total: 0, current: '' };
  renderLibrary();
  try {
    await fetch('/api/reclassify-all', { method: 'POST' });
    const poll = setInterval(async () => {
      try {
        const r = await fetch('/api/categorize-progress');
        const p = await r.json();
        _catProgress = p;
        renderLibrary();
        if (!p.running && p.total > 0 && p.done >= p.total) {
          clearInterval(poll);
          await loadCollections();
          _categorizingAll = false;
          _catProgress = null;
          renderLibrary();
        }
      } catch {}
    }, 1500);
  } catch {
    _categorizingAll = false;
    _catProgress = null;
    renderLibrary();
  }
}

async function categorizeAll() {
  if (_categorizingAll) return;
  _categorizingAll = true;
  renderLibrary();
  try {
    await fetch('/api/categorize-all', { method: 'POST' });
    // 완료까지 폴링 (3초 간격, 최대 5분)
    let attempts = 0;
    const poll = setInterval(async () => {
      attempts++;
      await loadCollections();
      const stillUncategorized = allCols.some(n => !sourceInfoCache[n]?.topic);
      if (!stillUncategorized || attempts > 100) {
        clearInterval(poll);
        _categorizingAll = false;
        renderLibrary();
      }
    }, 3000);
  } catch {
    _categorizingAll = false;
    renderLibrary();
  }
}

function renderLibCard(name) {
  const info    = sourceInfoCache[name] || {};
  const title   = info.title || name;
  const summary = info.summary || '';
  const dtype   = info.data_type || 'video';
  const segs    = info.segment_count ?? null;
  const dur     = info.duration_seconds != null ? fmtDuration(info.duration_seconds) : null;
  const sel     = libSelected.has(name);

  let badgeClass = 'lib-badge-video';
  let badgeLabel = '영상';
  if (dtype === 'experimental') { badgeClass = 'lib-badge-data'; badgeLabel = '데이터'; }
  else if (dtype === 'document') { badgeClass = 'lib-badge-doc'; badgeLabel = '문서'; }

  const meta = [
    segs != null ? `${segs}개 세그먼트` : null,
    dur ? dur : null,
  ].filter(Boolean).join(' · ');

  return `
  <div class="lib-card ${sel ? 'selected' : ''}" id="lib-card-${CSS.escape(name)}" onclick="toggleLibCard('${name}')">
    <div class="lib-card-top">
      <span class="lib-badge ${badgeClass}">${badgeLabel}</span>
      <div class="lib-card-check" aria-hidden="true"></div>
    </div>
    <div class="lib-card-title" title="${title}">${title}</div>
    ${(() => {
      const tags = (info.topics && info.topics.length ? info.topics : (info.topic ? [info.topic] : []));
      if (!tags.length) return '';
      return `<div class="lib-card-tags" style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0">` +
        tags.map(t => `<span style="font-size:10px;padding:2px 6px;border-radius:8px;background:#1c2840;color:#7aa9ff">${t}</span>`).join('') +
        `</div>`;
    })()}
    ${summary ? `<div class="lib-card-preview">${summary}</div>` : ''}
    ${meta ? `<div class="lib-card-meta">${meta}</div>` : ''}
    <div class="lib-card-actions">
      <button onclick="openInfoPanel(event,'${name}')" title="상세 정보">
        ${ic('doc')} 상세 정보
      </button>
      <button onclick="libOpenChat(event,'${name}')" title="단독 대화">
        ${ic('chat')} 대화
      </button>
    </div>
  </div>`;
}

function toggleLibCard(name) {
  if (libSelected.has(name)) libSelected.delete(name);
  else libSelected.add(name);
  const card = document.getElementById(`lib-card-${CSS.escape(name)}`);
  if (card) card.classList.toggle('selected', libSelected.has(name));
  updateLibSelBar();
}

function updateLibSelBar() {
  const count = libSelected.size;
  const countEl = document.getElementById('lib-sel-count');
  const chatBtn  = document.getElementById('lib-chat-btn');
  if (countEl) countEl.textContent = `${count}개 선택됨`;
  if (chatBtn) chatBtn.disabled = count === 0;
}

function libSelectAll() {
  allCols.forEach(n => libSelected.add(n));
  renderLibrary();
}

function libDeselectAll() {
  libSelected.clear();
  renderLibrary();
}

function libStartChat() {
  if (!libSelected.size) return;
  selectedCols.clear();
  libSelected.forEach(n => selectedCols.add(n));
  saveSelectedCols();
  renderCollections();
  switchMainTab('chat');
  addSystemMsg(`${libSelected.size}개 소스가 선택되었습니다. 질문하세요.`);
}

function libOpenChat(event, name) {
  event.stopPropagation();
  selectedCols.clear();
  selectedCols.add(name);
  saveSelectedCols();
  renderCollections();
  switchMainTab('chat');
  addSystemMsg(`"${name}" 소스로 대화를 시작합니다. 질문하세요.`);
}

// ══════════════════════════════════════════════════════════
//  소스 정보 패널
// ══════════════════════════════════════════════════════════
async function openInfoPanel(event, stem) {
  event.stopPropagation();
  const panel   = document.getElementById('info-panel');
  const overlay = document.getElementById('info-overlay');
  if (!panel) return;

  document.getElementById('info-panel-title').textContent = stem;
  document.getElementById('info-summary').textContent = '불러오는 중...';
  document.getElementById('info-files').innerHTML = '';
  document.getElementById('info-delete-btn').onclick = () => deleteCollection(stem);
  panel.classList.add('open');
  overlay.classList.add('open');

  let info = sourceInfoCache[stem];
  if (!info) {
    try {
      const res = await fetch(`/api/source-info/${encodeURIComponent(stem)}`);
      info = await res.json();
      sourceInfoCache[stem] = info;
    } catch { info = { title: stem, summary: '정보를 불러올 수 없습니다.' }; }
  }
  document.getElementById('info-panel-title').textContent = info.title || stem;
  document.getElementById('info-summary').textContent = info.summary || '요약 없음';

  try {
    const res   = await fetch(`/api/output-info/${encodeURIComponent(stem)}`);
    const files = await res.json();
    const fileEl = document.getElementById('info-files');
    fileEl.innerHTML = files.length
      ? files.map(f => {
          const icon = /\.mp4$/i.test(f) ? ic('film') : /\.srt$/i.test(f) ? ic('memo') : ic('doc');
          const isVid = /\.mp4$/i.test(f);
          return `<a class="info-file-item"
            href="/api/output/${encodeURIComponent(stem)}/${encodeURIComponent(f)}"
            ${isVid ? `onclick="showVideoPlayer('${stem}','${f}',event)"` : 'download'}>
            <span>${icon}</span>
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${f}</span>
          </a>`;
        }).join('')
      : '<div style="font-size:12px;color:var(--muted)">결과 파일이 없습니다.</div>';
  } catch {}
}

function closeInfoPanel() {
  document.getElementById('info-panel')?.classList.remove('open');
  document.getElementById('info-overlay')?.classList.remove('open');
}

// ══════════════════════════════════════════════════════════
//  RAG 임계값 슬라이더
// ══════════════════════════════════════════════════════════
const THRESHOLD_LABELS = {
  0: { val: 0.30, label: '정밀',   hint: '가장 엄격 — 완전히 일치하는 내용만' },
  1: { val: 0.45, label: '엄격',   hint: '엄격 — 관련성 높은 내용 위주' },
  2: { val: 0.60, label: '보통',   hint: '균형 잡힌 검색 (기본값)' },
  3: { val: 0.75, label: '넓게',   hint: '넓게 — 간접적으로 관련된 내용도 포함' },
  4: { val: 0.90, label: '매우 넓게', hint: '가장 넓음 — 주제가 느슨하게 연관된 내용까지' },
};

function updateThreshold(idx) {
  idx = parseInt(idx, 10);
  const entry = THRESHOLD_LABELS[idx] || THRESHOLD_LABELS[2];
  ragThreshold = entry.val;
  const display = document.getElementById('threshold-val');
  const hint    = document.getElementById('threshold-hint');
  const ticks   = document.querySelectorAll('#threshold-ticks span');
  const slider  = document.querySelector('.threshold-slider');
  if (display) display.textContent = entry.label;
  if (hint)    hint.textContent    = entry.hint;
  if (slider)  slider.setAttribute('aria-valuetext', entry.label);
  ticks.forEach((el, i) => el.classList.toggle('active', i === idx));
}

// ══════════════════════════════════════════════════════════
//  처리 옵션
// ══════════════════════════════════════════════════════════
function toggleTts() { /* TTS 비활성화 — 준비 중 */ }
function setLang(val) { targetLang = val; }


async function processYoutube(url) {
  const req = { url, target_lang: targetLang, skip_tts: true };
  let jobId;
  try {
    const res  = await fetch('/api/youtube', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    jobId = (await res.json()).jobId;
    activeJobId = jobId;
  } catch (e) { addMessage('ai', 'YouTube 처리 시작 실패: ' + e.message); return; }

  const bannerId = 'banner-' + jobId;
  addProcessBanner(bannerId, 'YouTube 영상', jobId);
  isProcessing = true;

  const es = new EventSource(`/api/logs/${jobId}`);
  es.addEventListener('message', e => {
    const data = JSON.parse(e.data);
    if (data.type === 'log') { updateBannerFromLog(bannerId, data.msg); detectLanguage(bannerId, data.msg); }
    if (data.type === 'done') {
      es.close(); isProcessing = false;
      if (data.status === 'done') {
        finishBanner(bannerId);
        loadCollections().then(() => {
          addMessage('ai', `${ic('check-ok')} YouTube 영상 처리 완료!\n\n이제 질문하세요.`);
          sendNotification('edu_pipeline', 'YouTube 영상 처리가 완료되었습니다.');
        });
      } else { failBanner(bannerId); addMessage('ai', 'YouTube 처리 중 오류가 발생했습니다.'); }
    }
  });
}

async function submitDoc() {
  const text = document.getElementById('doc-textarea')?.value.trim();
  if (!text) return;
  closeDocModal();
  hideWelcome();
  addMessage('user', `${ic('doc')} 문서 처리 시작 (${text.length}자)`);

  let jobId;
  try {
    const res  = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: text, file_type: 'document',
        target_lang: targetLang, skip_tts: true, skip_summary: false }),
    });
    jobId = (await res.json()).jobId;
    activeJobId = jobId;
  } catch (e) { addMessage('ai', '처리 시작 실패: ' + e.message); return; }

  const bannerId = 'banner-' + jobId;
  addProcessBanner(bannerId, '문서 텍스트', jobId);
  isProcessing = true;

  const es = new EventSource(`/api/logs/${jobId}`);
  es.addEventListener('message', e => {
    const data = JSON.parse(e.data);
    if (data.type === 'log') updateBannerFromLog(bannerId, data.msg);
    if (data.type === 'done') {
      es.close(); isProcessing = false;
      if (data.status === 'done') {
        finishBanner(bannerId);
        loadCollections().then(() => addMessage('ai', `✅ 문서 처리 완료! 이제 질문하세요.`));
        sendNotification('edu_pipeline', '문서 처리가 완료되었습니다.');
      } else { failBanner(bannerId); addMessage('ai', '문서 처리 중 오류가 발생했습니다.'); }
    }
  });
}

function fileIcon(name) {
  if (/\.(mp4|mkv|avi|mov|webm)$/i.test(name)) return ic('film');
  if (/\.pdf$/i.test(name)) return ic('doc');
  return ic('memo');
}

// ══════════════════════════════════════════════════════════
//  메시지 전송 (채팅 전용 — 파일 처리는 소스 패널에서)
// ══════════════════════════════════════════════════════════
async function handleSend() {
  const input = document.getElementById('msg-input');
  const text  = input.value.trim();
  if (!text) return;
  input.value = ''; autoResize(input); hideWelcome();
  await sendChat(text);
}

// ══════════════════════════════════════════════════════════
//  업로드 + 처리
// ══════════════════════════════════════════════════════════
async function uploadAndProcess(file, name, userMsg) {
  if (userMsg) addMessage('user', userMsg);
  else addMessage('user', `${fileIcon(name)} **${name}** 처리 시작`);

  addSystemMsg('파일 업로드 중...');
  const fd = new FormData(); fd.append('file', file);
  let filePath;
  try {
    filePath = (await (await fetch('/api/upload', { method: 'POST', body: fd })).json()).path;
  } catch (e) { addMessage('ai', '업로드 실패: ' + e.message); return; }

  const isVideo = /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(name);
  const stem    = name.replace(/\.[^.]+$/, '');
  let jobId;
  try {
    const res = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: filePath, file_type: isVideo ? 'video' : 'document',
        target_lang: targetLang, skip_tts: true, skip_summary: false }),
    });
    jobId = (await res.json()).jobId; activeJobId = jobId;
  } catch (e) { addMessage('ai', '처리 시작 실패: ' + e.message); return; }

  const bannerId = 'banner-' + jobId;
  addProcessBanner(bannerId, name, jobId);
  isProcessing = true;

  const es = new EventSource(`/api/logs/${jobId}`);
  es.addEventListener('message', async e => {
    const data = JSON.parse(e.data);
    if (data.type === 'log') { updateBannerFromLog(bannerId, data.msg); detectLanguage(bannerId, data.msg); }
    if (data.type === 'done') {
      es.close(); isProcessing = false;
      if (data.status === 'done') {
        finishBanner(bannerId);
        await loadCollections();
        selectedCols.add(stem); renderCollections();
        addMessage('ai', `${ic('check-ok')} **${name}** 처리 완료!\n\n이제 내용에 대해 질문하세요.`);
        await addDownloadButtons(stem);
        sendNotification('edu_pipeline', `${name} 처리가 완료되었습니다.`);
      } else { failBanner(bannerId); addMessage('ai', '처리 중 오류가 발생했습니다.'); }
    }
  });
}

// ══════════════════════════════════════════════════════════
//  다운로드 버튼
// ══════════════════════════════════════════════════════════
async function addDownloadButtons(stem) {
  let files;
  try {
    files = await (await fetch(`/api/output-info/${encodeURIComponent(stem)}`)).json();
  } catch { return; }
  if (!files?.length) return;

  const area = document.getElementById('chat-area');
  const wrap = document.createElement('div');
  wrap.style.cssText = 'padding: 0 max(calc(50% - 390px), 32px) 8px;';
  wrap.innerHTML = '<div class="dl-row">' +
    files.map(f => {
      const icon  = /\.mp4$/i.test(f) ? ic('film') : /\.srt$/i.test(f) ? ic('memo') : ic('doc');
      const label = /\.mp4$/i.test(f) ? '더빙 영상' : /\.srt$/i.test(f) ? '자막' : f;
      const isVid = /\.mp4$/i.test(f);
      return `<a class="dl-btn" href="/api/output/${encodeURIComponent(stem)}/${encodeURIComponent(f)}"
        ${isVid ? `onclick="showVideoPlayer('${stem}','${f}',event)"` : 'download'}>
        ${icon} ${label}
      </a>`;
    }).join('') + '</div>';
  area.appendChild(wrap);
  scrollBottom();
}

// ══════════════════════════════════════════════════════════
//  인라인 비디오 플레이어
// ══════════════════════════════════════════════════════════
function showVideoPlayer(stem, filename, event) {
  if (event) event.preventDefault();
  activeVideoStem = stem;
  document.getElementById('video-player-wrap')?.remove();
  const area = document.getElementById('chat-area');
  const wrap = document.createElement('div');
  wrap.className = 'video-wrap'; wrap.id = 'video-player-wrap';
  wrap.innerHTML = `
    <video id="main-video" controls preload="metadata">
      <source src="/api/output/${encodeURIComponent(stem)}/${encodeURIComponent(filename)}" type="video/mp4">
    </video>
    <div class="video-bar">
      <span>${ic('film')} ${stem} · 타임스탬프 클릭 시 해당 구간으로 이동</span>
      <button class="video-close" onclick="hideVideoPlayer()">${ic('close')}</button>
    </div>`;
  area.insertBefore(wrap, area.firstChild);
  scrollBottom();
}

function hideVideoPlayer() {
  document.getElementById('video-player-wrap')?.remove();
  activeVideoStem = null;
}

function seekVideo(seconds) {
  const v = document.getElementById('main-video'); if (!v) return;
  v.currentTime = seconds; v.play();
  document.getElementById('video-player-wrap')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function tsToSeconds(ts) {
  const parts = ts.split(':').map(Number);
  return parts.length === 3 ? parts[0]*3600+parts[1]*60+parts[2] : parts[0]*60+parts[1];
}

// ══════════════════════════════════════════════════════════
//  진행 배너
// ══════════════════════════════════════════════════════════
const STEPS    = ['오디오 추출', 'STT', '번역', 'TTS', '임베딩'];
const STEP_PAT = [
  /STEP 1|오디오|audio/i,
  /STEP 2|STT|음성 인식|transcri/i,
  /STEP 3|번역|translat/i,
  /STEP 4|TTS|더빙/i,
  /STEP 5|임베딩|embed/i,
];
const STT_PROG_RE = /(\d+)\/(\d+)|세그먼트\s+(\d+)[^\d]+(\d+)/;

function addProcessBanner(id, filename, jobId) {
  const el = document.createElement('div');
  el.className = 'process-banner'; el.id = id;
  el.innerHTML = `
    <div class="pb-title" style="display:flex;align-items:center">
      <span>🔄 처리 중: ${filename}</span>
      <span id="${id}-lang" style="font-size:11px;color:var(--muted);margin-left:8px"></span>
      <button class="stop-btn" onclick="stopJob('${jobId}')">■ 중지</button>
    </div>
    <div class="process-steps">
      ${STEPS.map((s,i) => `<div class="ps" id="${id}-s${i}">${s}</div>`).join('')}
    </div>`;
  document.getElementById('chat-area').appendChild(el);
  scrollBottom();
}

function updateBannerFromLog(id, msg) {
  STEP_PAT.forEach((pat, i) => {
    if (!pat.test(msg)) return;
    for (let j=0; j<i; j++) {
      const p = document.getElementById(`${id}-s${j}`);
      if (p) { p.classList.remove('active'); p.classList.add('done'); }
    }
    const el = document.getElementById(`${id}-s${i}`);
    if (el && !el.classList.contains('done')) {
      el.classList.add('active');
      if (i === 1) { // STT 진행률
        const m = msg.match(STT_PROG_RE);
        if (m) {
          const cur = m[1]||m[3], tot = m[2]||m[4];
          let prog = el.querySelector('.stt-prog');
          if (!prog) { prog=document.createElement('span'); prog.className='stt-prog'; el.appendChild(prog); }
          prog.textContent = `${cur}/${tot}`;
        }
      }
    }
  });
}

function detectLanguage(id, msg) {
  const m = msg.match(/detect|lang(?:uage)?[^\w]+([\w\-]+)/i) || msg.match(/언어[^\w]+([\w\-]+)/i);
  if (m) {
    const el = document.getElementById(`${id}-lang`);
    if (el) el.textContent = `(감지: ${m[1]})`;
  }
}

function finishBanner(id) {
  const b = document.getElementById(id); if (!b) return;
  STEPS.forEach((_, i) => {
    const el = document.getElementById(`${id}-s${i}`);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
  b.querySelector('.pb-title').innerHTML = '✅ 처리 완료!';
}
function failBanner(id) {
  const b = document.getElementById(id);
  if (b) b.querySelector('.pb-title').innerHTML = '❌ 처리 실패';
}

async function stopJob(jobId) {
  try {
    await fetch(`/api/stop/${jobId}`, { method: 'POST' });
    isProcessing = false;
    addSystemMsg('처리가 중지되었습니다.');
  } catch {}
}

// ══════════════════════════════════════════════════════════
//  스트리밍 채팅
// ══════════════════════════════════════════════════════════
const NO_SOURCE_SIGNAL = '업로드된 자료에서 해당 내용을 찾지 못했습니다';

async function sendChat(question) {
  if (isChatting) return;
  lastQuestion = question;
  addMessage('user', question);
  saveSession();

  const cols = selectedCols.size > 0 ? [...selectedCols] : [];
  if (!allCols.length && !cols.length) {
    addMessage('ai', '아직 처리된 파일이 없습니다. 파일을 첨부해 주세요!'); return;
  }

  isChatting = true;
  document.getElementById('send-btn').disabled = true;
  const thinkId = addThinking();

  let collectedChunks = [];

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, collections: cols, threshold: ragThreshold }),
    });
    removeThinking(thinkId);
    const { el: bubbleEl, row: rowEl } = addStreamingBubble();

    const STATUS_LABELS = {
      searching:   '자료 검색 중...',
      loading_llm: 'AI 모델 준비 중 (첫 실행은 시간이 걸립니다)...',
      generating:  '답변 생성 중...',
    };

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', fullText = '';

    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let payload; try { payload = JSON.parse(line.slice(6)); } catch { continue; }
        if (payload.status) {
          const label = STATUS_LABELS[payload.status] || '';
          if (label) bubbleEl.innerHTML = `<span class="status-hint">${label}</span>`;
          continue;
        }
        if (payload.chunks) { collectedChunks = payload.chunks; continue; }
        if (payload.token) {
          fullText += payload.token;
          bubbleEl.innerHTML = renderBubbleText(fullText) + '<span class="stream-cursor"></span>';
          scrollBottom();
        }
        if (payload.done || payload.error) break;
      }
    }

    if (fullText.includes(NO_SOURCE_SIGNAL)) {
      rowEl.remove(); addNoSourceMessage(fullText);
      if (currentSession) currentSession.messages.push({ role: 'ai', text: fullText, html: '', chunks: collectedChunks });
    } else {
      bubbleEl.innerHTML = renderBubbleText(fullText);
      addMessageActions(rowEl, bubbleEl, fullText);
      if (collectedChunks.length) addChunksReveal(rowEl, collectedChunks);
      addFollowUpChips(question, fullText);
      if (currentSession) currentSession.messages.push({ role: 'ai', text: fullText, html: bubbleEl.innerHTML, chunks: collectedChunks });
    }
    saveSession();

  } catch (e) {
    removeThinking(thinkId);
    addMessage('ai', '오류가 발생했습니다: ' + e.message);
  } finally {
    isChatting = false;
    document.getElementById('send-btn').disabled = false;
  }
}

async function regenerate() {
  if (!lastQuestion || isChatting) return;
  hideWelcome();
  await sendChat(lastQuestion);
}

// ══════════════════════════════════════════════════════════
//  메시지 액션 버튼 (복사 / 재생성)
// ══════════════════════════════════════════════════════════
function addMessageActions(rowEl, bubbleEl, text) {
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  actions.innerHTML = `
    <button class="action-btn" onclick="copyMsg(this,'${encodeURIComponent(text)}')">📋 복사</button>
    <button class="action-btn" onclick="regenerate()">🔄 재생성</button>`;
  rowEl.appendChild(actions);
}

function copyMsg(btn, encodedText) {
  const text = decodeURIComponent(encodedText);
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = `${ic('check')} 복사됨`; btn.classList.add('copied');
    setTimeout(() => { btn.innerHTML = `${ic('copy')} 복사`; btn.classList.remove('copied'); }, 2000);
  });
}

// ══════════════════════════════════════════════════════════
//  소스 청크 리뷰 (출처 펼치기)
// ══════════════════════════════════════════════════════════
function addChunksReveal(rowEl, chunks) {
  const bubble = rowEl.querySelector('.bubble'); if (!bubble) return;
  const reveal = document.createElement('div');
  reveal.className = 'chunks-reveal';
  const listId = 'chunks-' + Date.now();
  reveal.innerHTML = `
    <button class="chunks-toggle" onclick="toggleChunks('${listId}')">
      ${ic('book')} 참고 자료 ${chunks.length}개 보기 ${ic('chev-d')}
    </button>
    <div class="chunks-list" id="${listId}" style="display:none">
      ${chunks.map(c => `
        <div class="chunk-item">
          <div class="chunk-meta">
            <strong>${c.source}</strong>
            <span class="ts-link" onclick="seekVideo(${tsToSeconds(c.ts)})">[${c.ts}]</span>
            <span class="chunk-dist">유사도 ${(1-c.dist).toFixed(2)}</span>
          </div>
          <div>${c.text}</div>
        </div>`).join('')}
    </div>`;
  bubble.appendChild(reveal);
}

function toggleChunks(id) {
  const el  = document.getElementById(id); if (!el) return;
  const btn = el.previousElementSibling;
  const open = el.style.display === 'none';
  el.style.display = open ? 'flex' : 'none';
  if (btn) btn.innerHTML = `${ic('book')} 참고 자료 보기 ${ic(open ? 'chev-u' : 'chev-d')}`;
}

// ══════════════════════════════════════════════════════════
//  follow-up 제안 칩
// ══════════════════════════════════════════════════════════
const FOLLOWUP_TEMPLATES = [
  '더 자세히 설명해줘', '예시를 들어줘', '핵심만 요약해줘',
  '관련된 다른 개념은?', '왜 중요한가요?', '쉽게 설명해줘',
  '반대 의견은?', '실제로 어떻게 적용하나요?'
];

function addFollowUpChips(question, answer) {
  const area = document.getElementById('chat-area');
  const wrap = document.createElement('div');
  wrap.className = 'follow-chips';

  // 응답 키워드 기반 + 일반 제안 혼합
  const keywords = answer.match(/[가-힣]{2,6}/g) || [];
  const unique   = [...new Set(keywords)].slice(0, 2);
  const suggestions = [
    ...unique.map(k => `"${k}"에 대해 더 알려줘`),
    ...FOLLOWUP_TEMPLATES.slice(0, 3 - unique.length),
  ].slice(0, 3);

  wrap.innerHTML = suggestions.map(s =>
    `<div class="follow-chip" onclick="setInput('${s.replace(/'/g,"\\'")}'">${ic('chat')} ${s}</div>`
  ).join('');
  area.appendChild(wrap);
  scrollBottom();
}

// ══════════════════════════════════════════════════════════
//  텍스트 렌더링 (타임스탬프 링크 포함)
// ══════════════════════════════════════════════════════════
function renderBubbleText(text) {
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\[([^\]]*?\d{1,2}:\d{2}(?::\d{2})?)\]/g, (match, inner) => {
    const tsMatch = inner.match(/(\d{1,2}:\d{2}(?::\d{2})?)$/);
    if (!tsMatch) return match;
    return `<span class="ts-link" onclick="seekVideo(${tsToSeconds(tsMatch[1])})" title="${tsMatch[1]}로 이동">[${inner}]</span>`;
  });
  return html.replace(/\n/g, '<br>');
}

function addStreamingBubble() {
  const area = document.getElementById('chat-area');
  const row  = document.createElement('div');
  row.className = 'msg-row ai';
  row.innerHTML = `<div class="avatar ai">${ic('ai')}</div><div class="bubble"></div>`;
  area.appendChild(row); scrollBottom();
  return { el: row.querySelector('.bubble'), row };
}

// ══════════════════════════════════════════════════════════
//  메시지 렌더링
// ══════════════════════════════════════════════════════════
function addMessage(role, text) {
  const area = document.getElementById('chat-area');
  const row  = document.createElement('div');
  row.className = `msg-row ${role}`;
  const avatar = role === 'ai' ? `<div class="avatar ai">${ic('ai')}</div>` : `<div class="avatar user">${ic('user')}</div>`;
  const html   = role === 'ai' ? renderBubbleText(text)
    : text.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
  row.innerHTML = `${avatar}<div class="bubble">${html}</div>`;
  area.appendChild(row); scrollBottom();
  if (currentSession) currentSession.messages.push({ role, text, html });
}

function addNoSourceMessage(text) {
  const area  = document.getElementById('chat-area');
  const row   = document.createElement('div');
  row.className = 'msg-row ai';
  const lines = text.split('\n').filter(l => l.trim());
  const title = lines[0] || '소스를 찾을 수 없습니다';
  const rest  = lines.slice(1).map(l =>
    l.startsWith('•') ? `<li>${l.slice(1).trim()}</li>` : `<p>${l}</p>`
  ).join('');
  row.innerHTML = `
    <div class="avatar ai">${ic('ai')}</div>
    <div class="bubble no-source">
      <div class="no-source-header">${ic('search')}<span>${title}</span></div>
      ${rest ? `<ul style="margin:6px 0 0 4px;padding-left:16px;font-size:13px;color:var(--text-2);line-height:1.9">${rest}</ul>` : ''}
    </div>`;
  area.appendChild(row); scrollBottom();
}

function addSystemMsg(text) {
  const area = document.getElementById('chat-area');
  const el   = document.createElement('div');
  el.style.cssText = 'text-align:center;font-size:12px;color:var(--muted);padding:8px';
  el.textContent = text;
  area.appendChild(el); scrollBottom();
}

function addThinking() {
  const area = document.getElementById('chat-area');
  const row  = document.createElement('div');
  const id   = 'think-' + Date.now();
  row.className = 'msg-row ai'; row.id = id;
  row.innerHTML = `<div class="avatar ai">${ic('ai')}</div>
    <div class="bubble thinking"><div class="dot-pulse"><span></span><span></span><span></span></div><span class="thinking-label">응답 준비 중</span></div>`;
  area.appendChild(row); scrollBottom();
  return id;
}
function removeThinking(id) { document.getElementById(id)?.remove(); }

// ══════════════════════════════════════════════════════════
//  대화 히스토리
// ══════════════════════════════════════════════════════════
function loadHistory() {
  try { sessions = JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch { sessions = []; }
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById('history-list'); if (!el) return;
  if (!sessions.length) {
    el.innerHTML = '<div class="empty-hint" style="font-size:11.5px">이전 대화가<br>없습니다.</div>';
    return;
  }
  el.innerHTML = sessions.slice().reverse().map(s => `
    <div class="history-item ${currentSession?.id===s.id?'active':''}"
         onclick="openSession('${s.id}')" title="${s.title}"
         style="display:flex;align-items:center;gap:6px">
      <span class="history-icon">${ic('chat')}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.title}</span>
      <span class="hist-del" title="삭제"
            onclick="event.stopPropagation();deleteSession('${s.id}')"
            style="opacity:.5;cursor:pointer;padding:0 4px;font-size:14px">✕</span>
    </div>`).join('');
}

function deleteSession(id) {
  if (!confirm('이 대화를 삭제할까요?')) return;
  sessions = sessions.filter(s => s.id !== id);
  try { localStorage.setItem(LS_KEY, JSON.stringify(sessions)); } catch {}
  if (currentSession?.id === id) {
    startNewSession();
    const area = document.getElementById('chat-area');
    if (area) area.innerHTML = '';
    newChat();
  }
  renderHistory();
}

function startNewSession() {
  currentSession = { id: Date.now().toString(), title: '새 대화', ts: Date.now(), messages: [] };
  renderHistory();
}

function saveSession() {
  if (!currentSession || !currentSession.messages.length) return;
  const first = currentSession.messages.find(m => m.role==='user');
  if (first) currentSession.title = first.text.slice(0,30) + (first.text.length>30?'…':'');
  const idx = sessions.findIndex(s => s.id===currentSession.id);
  if (idx>=0) sessions[idx]={...currentSession}; else sessions.push({...currentSession});
  if (sessions.length>30) sessions=sessions.slice(-30);
  try { localStorage.setItem(LS_KEY, JSON.stringify(sessions)); } catch {}
  renderHistory();
}

function openSession(id) {
  const s = sessions.find(s=>s.id===id); if (!s) return;
  saveSession();
  currentSession = { ...s, messages: [...s.messages] };
  const area = document.getElementById('chat-area');
  area.innerHTML=''; hideVideoPlayer();
  s.messages.forEach(m => {
    const row = document.createElement('div');
    row.className = `msg-row ${m.role}`;
    const avatar = m.role==='ai'?`<div class="avatar ai">${ic('ai')}</div>`:`<div class="avatar user">${ic('user')}</div>`;
    row.innerHTML = `${avatar}<div class="bubble">${m.html||renderBubbleText(m.text||'')}</div>`;
    area.appendChild(row);
  });
  scrollBottom(); renderHistory();
}

// ══════════════════════════════════════════════════════════
//  UI 헬퍼
// ══════════════════════════════════════════════════════════
function hideWelcome() { document.getElementById('welcome')?.remove(); }
function scrollBottom() { const a=document.getElementById('chat-area'); a.scrollTop=a.scrollHeight; }
function onKey(e) { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();} }
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,160)+'px'; }
function setInput(text) { const el=document.getElementById('msg-input'); el.value=text; el.focus(); autoResize(el); }

function newChat() {
  saveSession(); startNewSession();
  document.getElementById('chat-area').innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome-icon">${ic('book')}</div>
      <h2>edu_pipeline에 오신 것을 환영합니다</h2>
      <p>영상이나 문서를 업로드하면 자동으로 처리하고,<br>내용을 바탕으로 질문에 답해드립니다.</p>
      <div class="hint-grid">
        <div class="hint-card" onclick="setInput('이 영상의 핵심 내용을 요약해줘')"><div class="hc-icon">${ic('memo')}</div><div class="hc-text">이 영상의 핵심 내용을 요약해줘</div></div>
        <div class="hint-card" onclick="setInput('어려운 개념을 쉽게 설명해줘')"><div class="hc-icon">${ic('bulb')}</div><div class="hc-text">어려운 개념을 쉽게 설명해줘</div></div>
        <div class="hint-card" onclick="setInput('중요한 키워드를 알려줘')"><div class="hc-icon">${ic('key')}</div><div class="hc-text">중요한 키워드를 알려줘</div></div>
        <div class="hint-card" onclick="setInput('퀴즈 문제를 만들어줘')"><div class="hc-icon">${ic('question')}</div><div class="hc-text">퀴즈 문제를 만들어줘</div></div>
      </div>
    </div>`;
  const inp = document.getElementById('msg-input');
  inp.value=''; autoResize(inp); hideVideoPlayer();
}

// ══════════════════════════════════════════════════════════
//  소스 추가 패널
// ══════════════════════════════════════════════════════════
let _aspFile      = null;   // 선택된 파일
let _aspJobId     = null;   // 진행 중인 job
let _aspStem      = null;   // 처리 완료된 컬렉션 stem
let _aspActiveTab = 'file';

const ASP_STEP_PAT = [
  /STEP 1|오디오|audio|문서 파싱/i,
  /STEP 2|STT|음성 인식|transcri/i,
  /STEP 3|번역|translat/i,
  /STEP 5|임베딩|embed/i,
  /STEP 6|인덱싱|chroma/i,
];

// ── 메인 탭 전환 ────────────────────────────────────────────────────
function switchMainTab(tab, subtab) {
  document.getElementById('tab-chat').style.display    = tab === 'chat'    ? 'flex'  : 'none';
  document.getElementById('tab-library').style.display = tab === 'library' ? 'flex'  : 'none';
  document.getElementById('tab-source').style.display  = tab === 'source'  ? 'block' : 'none';
  ['chat', 'library', 'source'].forEach(t => {
    const btn = document.getElementById(`main-tab-${t}`);
    btn?.classList.toggle('active', t === tab);
    btn?.setAttribute('aria-selected', t === tab ? 'true' : 'false');
  });
  if (tab === 'library') {
    renderLibrary();
  }
  if (tab === 'source') {
    if (subtab) {
      const btn = document.getElementById(`asp-tab-btn-${subtab}`);
      if (btn) switchAspTab(subtab, btn);
    }
  }
}

// openSourcePanel — 이제 탭 전환으로 처리
function openSourcePanel(tab) {
  switchMainTab('source', tab || 'file');
}

function closeSourcePanel() {
  switchMainTab('chat');
}

// 소스 탭 드래그 앤 드롭
function sourceDragOver(e) {
  e.preventDefault();
  document.querySelector('.source-pane')?.classList.add('drag-over');
}
function sourceDragLeave(e) {
  if (!document.querySelector('.source-pane')?.contains(e.relatedTarget)) {
    document.querySelector('.source-pane')?.classList.remove('drag-over');
  }
}
function sourceDrop(e) {
  e.preventDefault();
  document.querySelector('.source-pane')?.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) { aspSetFile(f); switchAspTab('file', document.getElementById('asp-tab-btn-file')); }
}

function closeSourcePanel() {
  document.getElementById('asp-overlay').classList.remove('open');
  document.getElementById('asp-panel').classList.remove('open');
}

function aspResetUI() {
  _aspFile  = null; _aspJobId = null; _aspStem = null;
  ['asp-progress','asp-result'].forEach(id => {
    const el = document.getElementById(id); if (el) el.style.display = 'none';
  });
  document.getElementById('asp-start-btn').style.display = '';
  document.getElementById('asp-file-preview').style.display = 'none';
  document.getElementById('asp-dropzone').style.display = '';
  const ui = document.getElementById('asp-url-input');
  const ti = document.getElementById('asp-text-input');
  if (ui) ui.value = '';
  if (ti) ti.value = '';
  switchAspTab('file', document.getElementById('asp-tab-btn-file'));
}

function switchAspTab(tab, btn) {
  _aspActiveTab = tab;
  ['file','youtube','text'].forEach(t => {
    const el = document.getElementById(`asp-tab-${t}`);
    if (el) el.style.display = t === tab ? '' : 'none';
  });
  document.querySelectorAll('.asp-tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

/* ── 드래그 앤 드롭 ── */
function aspDragOver(e) { e.preventDefault(); document.getElementById('asp-dropzone').classList.add('drag-over'); }
function aspDragLeave()  { document.getElementById('asp-dropzone').classList.remove('drag-over'); }
function aspDrop(e) {
  e.preventDefault(); aspDragLeave();
  const f = e.dataTransfer.files[0]; if (f) aspSetFile(f);
}
function aspFileChosen(input) { if (input.files[0]) aspSetFile(input.files[0]); input.value=''; }

function aspSetFile(file) {
  _aspFile = file;
  const isVideo = /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(file.name);
  document.getElementById('asp-dropzone').style.display  = 'none';
  document.getElementById('asp-file-preview').style.display = '';
  document.getElementById('asp-file-icon').innerHTML = ic(isVideo ? 'film' : 'doc');
  document.getElementById('asp-file-name').textContent  = file.name;
  document.getElementById('asp-file-size').textContent  = (file.size / 1024 / 1024).toFixed(1) + ' MB';
}
function aspClearFile() {
  _aspFile = null;
  document.getElementById('asp-file-preview').style.display = 'none';
  document.getElementById('asp-dropzone').style.display = '';
}

/* ── 처리 시작 ── */
async function startAspProcess() {
  const lang = document.getElementById('asp-lang-select')?.value || 'ko';
  let stem;

  document.getElementById('asp-start-btn').style.display = 'none';
  document.getElementById('asp-progress').style.display  = '';
  _aspResetSteps();

  let jobId;
  try {
    if (_aspActiveTab === 'file') {
      if (!_aspFile) { _aspShowError('파일을 선택해 주세요.'); return; }
      // 업로드
      const fd = new FormData(); fd.append('file', _aspFile);
      const upRes  = await fetch('/api/upload', { method: 'POST', body: fd });
      const upData = await upRes.json();
      stem = _aspFile.name.replace(/\.[^.]+$/, '');
      const isVideo = /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(_aspFile.name);
      const res = await fetch('/api/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: upData.path, file_type: isVideo ? 'video' : 'document',
          target_lang: lang, skip_tts: true, skip_summary: false }),
      });
      jobId = (await res.json()).jobId;

    } else if (_aspActiveTab === 'youtube') {
      const url = document.getElementById('asp-url-input')?.value.trim();
      if (!url) { _aspShowError('YouTube URL을 입력해 주세요.'); return; }
      const res = await fetch('/api/youtube', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, target_lang: lang }),
      });
      const data = await res.json();
      jobId = data.jobId; stem = 'youtube_video';   // stem은 완료 로그에서 파싱

    } else {
      const text = document.getElementById('asp-text-input')?.value.trim();
      if (!text) { _aspShowError('텍스트를 입력해 주세요.'); return; }
      const res = await fetch('/api/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: text, file_type: 'document',
          target_lang: lang, skip_tts: true, skip_summary: false }),
      });
      jobId = (await res.json()).jobId; stem = 'document';
    }
  } catch (e) { _aspShowError('요청 실패: ' + e.message); return; }

  _aspJobId = jobId;
  document.getElementById('asp-stop-btn').style.display = '';

  // SSE 스트리밍
  const es = new EventSource(`/api/logs/${jobId}`);
  es.addEventListener('message', async ev => {
    const data = JSON.parse(ev.data);
    if (data.type === 'log') {
      _aspUpdateSteps(data.msg);
      // 완료 로그에서 출력 디렉토리 stem 파싱 (예: "출력 디렉토리: data/output/XXX")
      const m = data.msg.match(/출력 디렉토리[:\s]+.*[/\\]([^/\\]+?)\s*$/);
      if (m) stem = m[1].trim();
    }
    if (data.type === 'error') {
      es.close();
      _aspShowError(data.msg || '서버 오류');
      return;
    }
    if (data.type === 'done') {
      es.close();
      document.getElementById('asp-stop-btn').style.display = 'none';
      if (data.status === 'done') {
        _aspMarkAllDone();
        _aspStem = stem;
        await loadCollections();
        if (stem) selectedCols.add(stem); renderCollections();
        await _aspShowResult(stem);
        sendNotification('edu_pipeline', `${stem} 처리가 완료되었습니다.`);
      } else {
        _aspShowError('처리 중 오류가 발생했습니다. 로그를 확인하세요.');
      }
    }
  });
}

function stopAspProcess() {
  if (_aspJobId) fetch(`/api/stop/${_aspJobId}`, { method: 'POST' });
  _aspJobId = null;
  document.getElementById('asp-stop-btn').style.display = 'none';
  document.getElementById('asp-progress').style.display = 'none';
  document.getElementById('asp-start-btn').style.display = '';
  _aspResetSteps();
}

/* ── 단계 업데이트 ── */
function _aspResetSteps() {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById(`asp-step-${i}`);
    if (el) el.classList.remove('active', 'done');
  }
  const titleEl = document.querySelector('#asp-progress .asp-progress-title');
  if (titleEl) { titleEl.textContent = '처리 중...'; titleEl.style.color = ''; }
  // 로그 박스 초기화
  const box = document.getElementById('asp-log-box');
  if (box) box.innerHTML = '';
  _setAspLogOpen(false);
}
function _aspUpdateSteps(line) {
  // 단계 업데이트
  for (let i = 0; i < ASP_STEP_PAT.length; i++) {
    if (ASP_STEP_PAT[i].test(line)) {
      for (let j = 0; j < i; j++) {
        const el = document.getElementById(`asp-step-${j}`);
        if (el) { el.classList.remove('active'); el.classList.add('done'); }
      }
      const cur = document.getElementById(`asp-step-${i}`);
      if (cur) { cur.classList.remove('done'); cur.classList.add('active'); }
      break;
    }
  }
  // 로그 박스에 라인 추가
  _aspAppendLog(line);
}

function _aspAppendLog(line) {
  const box = document.getElementById('asp-log-box');
  if (!box) return;
  const text = line.trim();
  if (!text) return;
  const row = document.createElement('div');
  row.className = 'asp-log-line' + (text.includes('[오류]') || text.includes('Error') || text.includes('error') ? ' asp-log-err' : '');
  row.textContent = text;
  box.appendChild(row);
  // 로그 박스 하단으로 자동 스크롤
  box.scrollTop = box.scrollHeight;
}

let _aspLogOpen = false;
function toggleAspLog() {
  _aspLogOpen = !_aspLogOpen;
  _setAspLogOpen(_aspLogOpen);
}
function _setAspLogOpen(open) {
  _aspLogOpen = open;
  const box    = document.getElementById('asp-log-box');
  const toggle = document.getElementById('asp-log-toggle');
  if (box)    box.style.display = open ? 'block' : 'none';
  if (toggle) {
    toggle.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="${open ? '18 15 12 9 6 15' : '6 9 12 15 18 9'}"/></svg> 로그 ${open ? '숨기기' : '보기'}`;
  }
  if (open) {
    const box2 = document.getElementById('asp-log-box');
    if (box2) box2.scrollTop = box2.scrollHeight;
  }
}

function _aspMarkAllDone() {
  for (let i = 0; i < 5; i++) {
    const el = document.getElementById(`asp-step-${i}`);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  }
}

function _aspShowError(msg) {
  const titleEl = document.querySelector('#asp-progress .asp-progress-title');
  if (titleEl) { titleEl.textContent = `오류: ${msg}`; titleEl.style.color = 'var(--red)'; }
  document.getElementById('asp-stop-btn').style.display = 'none';
  document.getElementById('asp-start-btn').style.display = '';
  // 에러 시 자동으로 로그 펼치기
  _setAspLogOpen(true);
}

/* ── 결과 표시 + 중복 감지 ── */
async function _aspShowResult(stem) {
  document.getElementById('asp-result').style.display = '';

  // 콘텐츠 미리보기
  try {
    const res  = await fetch(`/api/content-preview/${encodeURIComponent(stem)}`);
    const data = await res.json();
    const meta = document.getElementById('asp-preview-meta');
    const txt  = document.getElementById('asp-preview-text');
    if (meta) {
      const dur = data.duration ? ` · ${Math.floor(data.duration/60)}분 ${data.duration%60}초` : '';
      meta.textContent = `세그먼트 ${data.count}개${dur}`;
    }
    if (txt) txt.textContent = data.preview || '미리보기를 불러올 수 없습니다.';
  } catch {}

  // 중복 감지
  try {
    const res  = await fetch(`/api/check-duplicate/${encodeURIComponent(stem)}`);
    const data = await res.json();
    const warn = document.getElementById('asp-dup-warn');
    const txt  = document.getElementById('asp-dup-text');
    if (data.duplicates?.length && warn && txt) {
      const top = data.duplicates[0];
      const others = data.duplicates.map(d => `"${d.stem}" (${d.similarity}%)`).join(', ');
      txt.innerHTML = `<strong>유사 콘텐츠가 감지되었습니다.</strong><br>
        기존 소스 ${others}와(과) 내용이 ${top.similarity}% 이상 겹칩니다.<br>
        중복 저장 시 검색 결과가 편향될 수 있습니다.`;
      warn.style.display = '';
    }
  } catch {}
}

function openChatFromAsp() {
  switchMainTab('chat');
  if (_aspStem) {
    selectedCols.clear(); selectedCols.add(_aspStem); renderCollections();
  }
  hideWelcome();
  addMessage('ai', `소스 **${_aspStem || ''}** 이(가) 준비되었습니다. 무엇이든 질문해 보세요.`);
}

