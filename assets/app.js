/* CFB Spread Model front end. Reads the JSON the daily job writes to /data. */
(function () {
  'use strict';

  const S = {};             // loaded data
  const UI = {
    tab: 'week', view: 'spread', minEdge: 0, conf: '', q: '',
    gameId: null, custom: null, resWeek: null, rsort: { k: 'power', d: -1 }, rconf: '', rq: '',
    wsort: { k: 'kickoff', d: 1 }
  };
  const KEY_LINES = new Set([3, 7, 10, 14, 17, 21, 24, 28]);
  const BREAK_EVEN = 52.4;

  // ---------- helpers ----------
  const $ = (s, el) => (el || document).querySelector(s);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const sgn = (x, d = 1) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
  const num = (x, d = 1) => (x == null || isNaN(x) ? '–' : Number(x).toFixed(d));
  const pct = (x, d = 0) => (x == null ? '–' : (x * 100).toFixed(d) + '%');
  const fmtLine = (x) => { if (x == null) return '–'; if (x === 0) return 'PK'; return (x > 0 ? '+' : '−') + trim(Math.abs(x)); };
  const trim = (x) => (Math.round(x * 10) / 10).toString();
  const abbr = (t) => t.abbr || t.school;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } }
  };
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  async function load(name) {
    const r = await fetch('data/' + name + '?v=' + Date.now());
    if (!r.ok) throw new Error(name + ' ' + r.status);
    return r.json();
  }

  function ctTime(iso, tbd) {
    if (!iso) return '–';
    const d = new Date(iso);
    const day = d.toLocaleDateString('en-US', { timeZone: 'America/Chicago', weekday: 'short', month: 'short', day: 'numeric' });
    if (tbd) return day + ', TBD';
    const t = d.toLocaleTimeString('en-US', { timeZone: 'America/Chicago', hour: 'numeric', minute: '2-digit' });
    return day + ' ' + t + ' CT';
  }
  function builtStamp(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { timeZone: 'America/Chicago', month: 'short', day: 'numeric' }) + ' ' +
      d.toLocaleTimeString('en-US', { timeZone: 'America/Chicago', hour: 'numeric', minute: '2-digit' });
  }

  // ---------- model math (mirrors the Python) ----------
  function overrides(g) { return store.get('cfb-ovr-' + g.id, { qbHome: 0, qbAway: 0, extras: [], useExtras: false, wx: null }); }
  function saveOverrides(g, o) { store.set('cfb-ovr-' + g.id, o); }

  const EXTRA_POS = [
    ['QB', 0], ['RB1', 0.5], ['WR1 / top target', 0.8], ['WR2', 0.4], ['TE1', 0.3], ['OL starter', 0.5],
    ['Edge rusher', 0.6], ['DL starter', 0.4], ['LB starter', 0.3], ['CB1', 0.6], ['Safety', 0.3], ['Kicker', 0.4]
  ];
  const STATUS_MULT = { Out: 1, Doubtful: 0.75, Questionable: 0.35 };

  function adjMargin(g) {
    const o = overrides(g);
    const qbPts = S.model.spread.qb_change_points;
    let m = g.model.margin;
    m += -qbPts * (o.qbHome ? 1 : 0) + qbPts * (o.qbAway ? 1 : 0);
    if (o.useExtras) {
      for (const e of o.extras) m += (e.side === 'home' ? -1 : 1) * e.val;
    }
    return m;
  }
  function weatherFor(g) {
    const o = overrides(g);
    if (o.wx) return o.wx;
    const w = g.weather || {};
    return { venue: g.dome ? 'dome' : 'outdoor', precip: (w.snow_in || 0) >= 0.1 ? 'snow' : ((w.precip_in || 0) >= 0.1 ? 'rain' : 'none'),
      wind: Math.round(w.wind_mph || 0), temp: Math.round(w.temp_f == null ? 60 : w.temp_f), forecast: !!g.weather };
  }
  function totalParts(g) {
    const W = S.model.total.weather;
    const wx = weatherFor(g);
    const o = overrides(g);
    const parts = [];
    const base = g.model.total;
    let domeAdj = 0;
    if (!g.dome && wx.venue === 'dome') domeAdj = S.model.total.coef[S.model.total.features.indexOf('dome_f')] || 0;
    if (g.dome && wx.venue !== 'dome') domeAdj = -(S.model.total.coef[S.model.total.features.indexOf('dome_f')] || 0);
    const out = wx.venue === 'dome';
    const wind = out ? 0 : W.wind_per_mph_over_10 * Math.max(0, wx.wind - 10);
    const cold = out ? 0 : (wx.temp < 32 ? W.cold_below_32 : 0);
    const rain = out ? 0 : (wx.precip === 'rain' ? W.rain : 0);
    const snow = out ? 0 : (wx.precip === 'snow' ? W.snow : 0);
    const qb = S.model.total.qb_change_points * ((o.qbHome ? 1 : 0) + (o.qbAway ? 1 : 0));
    parts.push(['Base (team scoring)', base], ['Dome', domeAdj], ['Wind', wind], ['Cold', cold], ['QB changes', qb],
      ['Rain / snow' + (W.fitted ? '' : ' (untested)'), rain + snow]);
    const total = parts.reduce((a, p) => a + p[1], 0);
    return { parts, total, wx };
  }
  function adjTotal(g) { return totalParts(g).total; }

  function spreadCalc(g) {
    if (!g.line || g.line.spread == null) return null;
    const m = adjMargin(g);
    const L = g.line.spread;                // home spread, negative = home favored
    const edge = m + L;                     // > 0: model likes home
    const homeSide = edge >= 0;
    const sideTeam = homeSide ? g.home : g.away;
    const sideLine = homeSide ? L : -L;
    const ae = Math.abs(edge);
    const p = Math.min(0.56, sigmoid(S.model.spread.cover_slope * ae));
    const push = Number.isInteger(sideLine) ? pushRate(Math.abs(sideLine)) : 0;
    return { m, edge, ae, homeSide, sideTeam, sideLine, p, push };
  }
  function totalCalc(g) {
    if (!g.line || g.line.total == null) return null;
    const t = adjTotal(g);
    const edge = t - g.line.total;
    const ae = Math.abs(edge);
    const p = Math.min(0.58, sigmoid(S.model.total.hit_slope * ae));
    const push = Number.isInteger(g.line.total) ? 0.02 : 0;
    return { t, edge, ae, over: edge >= 0, p, push };
  }
  function pushRate(k) { const r = S.model.push_rate || {}; return r[String(Math.round(k))] || 0; }
  function flagOf(ae) { return ae >= 7 ? 'f7' : ae >= 5 ? 'f5' : ''; }
  function flagHtml(ae) {
    if (ae >= 7) return '<span class="flag f7">7+ EDGE</span>';
    if (ae >= 5) return '<span class="flag f5">5+ EDGE</span>';
    return '';
  }

  // custom matchup from team ratings (stats-only model: no FPI for made-up games)
  function customGame(homeId, awayId, neutral) {
    const R = S.ratings, T = (id) => R.teams.find((t) => t.id === +id);
    const h = T(homeId), a = T(awayId);
    if (!h || !a) return null;
    const sp = S.model.spread;
    const wk = S.site.week;
    const b = wk <= 3 ? 0 : wk <= 6 ? 1 : 2;
    const co = sp.fallback_coef[String(b)];
    const x = {};
    for (const m of ['sr', 'ypp', 'epa', 'pts']) x['x_' + m] = (h.r[m + '_o'] + a.r[m + '_d']) - (a.r[m + '_o'] + h.r[m + '_d']);
    x.loc = neutral ? 0 : 1;
    x.fluck_diff = 0;
    const comps = []; let margin = co[co.length - 1];
    const labels = { x_sr: 'Success rate edge', x_ypp: 'Yards per play edge', x_epa: 'EPA per play edge', x_pts: 'Points edge', loc: 'Home field', fluck_diff: 'Fumble luck (regressed)' };
    sp.fallback_features.forEach((f, i) => { const v = co[i] * (x[f] || 0); margin += v; comps.push({ label: labels[f] || f, pts: v }); });
    comps.push({ label: 'Baseline', pts: co[co.length - 1] });
    // total
    const tm = S.model.total, mu = R.mu;
    const xp = (mu.pts + h.r.pts_o + a.r.pts_d) + (mu.pts + a.r.pts_o + h.r.pts_d);
    const pace = mu.pace + h.r.pace_o + a.r.pace_o + h.r.pace_d + a.r.pace_d;
    const te = (h.r.epa_o + a.r.epa_d) + (a.r.epa_o + h.r.epa_d);
    const tv = { xp_sum: xp, x_pace: pace, x_tot_eff: te, dome_f: 0, env: R.env };
    let total = tm.coef[tm.coef.length - 1] + tm.median_offset;
    tm.features.forEach((f, i) => { total += tm.coef[i] * tv[f]; });
    const mk = (t) => ({ id: t.id, school: t.school, abbr: t.abbr, rank: null, qb: { primary: t.qb, last: t.qb_last, changed: t.qb_changed } });
    return { id: 'custom-' + homeId + '-' + awayId + (neutral ? 'n' : ''), custom: true, wk, kickoff: null, neutral: !!neutral, dome: false,
      home: mk(h), away: mk(a), line: null, fpi: null, weather: null, situ: {},
      model: { margin, total, comps, tcomps: [], uses_fpi: false }, final: false };
  }

  // ---------- header ----------
  function renderHeader() {
    const s = S.site;
    $('#sub').textContent = `${s.season} season · ratings through week ${s.ratings_through} · FBS play-by-play · built ${builtStamp(s.built)}`;
    const games = S.week.games.filter((g) => !g.final);
    const calcs = games.map(spreadCalc).filter(Boolean);
    const n7 = calcs.filter((c) => c.ae >= 7).length, n5 = calcs.filter((c) => c.ae >= 5).length;
    const t = s.ats; const t5 = s.ats5; const t7 = s.ats7;
    const lineSrc = s.dk_lines ? 'DraftKings' : 'DraftKings (not posted yet)';
    $('#chips').innerHTML = `
      <span class="chip">Week ${s.week}</span>
      <span class="chip g5">${n5} flagged at 5+</span>
      <span class="chip g7">${n7} at 7+</span>
      <span class="chip">Season ATS ${t.w}-${t.l}</span>
      <span class="chip">5+ ${t5.w}-${t5.l} · 7+ ${t7.w}-${t7.l}</span>
      <span class="chip">Lines: ${lineSrc}</span>
      <span class="spacer"></span>
      <button class="btn" id="howBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h10M4 18h7"/></svg>How it works</button>
      <button class="btn" id="themeBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg><span id="themeLbl"></span></button>`;
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    $('#themeLbl').textContent = light ? 'Dark mode' : 'Light mode';
    $('#themeBtn').onclick = () => {
      const now = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', now);
      store.set('cfb-theme-obj', now);
      try { localStorage.setItem('cfb-theme', now); } catch (e) { /* blocked */ }
      renderHeader();
    };
    $('#howBtn').onclick = () => setTab('method');
  }

  // ---------- tabs ----------
  function setTab(t) {
    UI.tab = t;
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
    for (const id of ['week', 'detail', 'results', 'ratings', 'method']) $('#tab-' + id).classList.toggle('hide', id !== t);
    render();
    try { history.replaceState(null, '', '#' + t + (t === 'detail' && UI.gameId ? '-' + UI.gameId : '')); } catch (e) { /* ignore */ }
  }
  function render() {
    renderHeader();
    if (UI.tab === 'week') renderWeek();
    if (UI.tab === 'detail') renderDetail();
    if (UI.tab === 'results') renderResults();
    if (UI.tab === 'ratings') renderRatings();
    if (UI.tab === 'method') renderMethod();
  }

  function matchup(g, opts = {}) {
    const rk = (t) => (t.rank ? `<span class="rk">${t.rank}</span>` : '');
    const tags = [];
    if (g.neutral) tags.push('<span class="tag">NEUTRAL</span>');
    if (g.home.qb && g.home.qb.changed) tags.push(`<span class="tag red">${esc(abbr(g.home))} QB CHANGE</span>`);
    if (g.away.qb && g.away.qb.changed) tags.push(`<span class="tag red">${esc(abbr(g.away))} QB CHANGE</span>`);
    const wx = g.weather;
    if (wx && !g.dome) {
      if ((wx.snow_in || 0) >= 0.1) tags.push('<span class="tag blue">SNOW</span>');
      else if ((wx.precip_in || 0) >= 0.1) tags.push('<span class="tag blue">RAIN</span>');
      if ((wx.wind_mph || 0) >= 15) tags.push(`<span class="tag blue">WIND ${Math.round(wx.wind_mph)}</span>`);
    }
    const o = overrides(g);
    if (o.qbHome || o.qbAway) tags.push('<span class="tag red">QB OVERRIDE</span>');
    return `<span class="mt">${rk(g.away)}${esc(g.away.school)}<span class="at">${g.neutral ? 'vs' : '@'}</span>${rk(g.home)}${esc(g.home.school)}</span>${opts.noTags ? '' : tags.join('')}`;
  }

  // ---------- THIS WEEK ----------
  function renderWeek() {
    const el = $('#tab-week');
    const W = S.week;
    const confs = [...new Set(W.games.flatMap((g) => [confOf(g.home.id), confOf(g.away.id)]).filter(Boolean))].sort();
    const isSpread = UI.view === 'spread';
    let rows = W.games.map((g) => ({ g, c: isSpread ? spreadCalc(g) : totalCalc(g) }));
    const q = UI.q.trim().toLowerCase();
    if (q) rows = rows.filter(({ g }) => (g.home.school + ' ' + g.away.school).toLowerCase().includes(q));
    if (UI.conf) rows = rows.filter(({ g }) => confOf(g.home.id) === UI.conf || confOf(g.away.id) === UI.conf);
    if (UI.minEdge) rows = rows.filter(({ c, g }) => c && !g.final && c.ae >= UI.minEdge);
    const k = UI.wsort.k, d = UI.wsort.d;
    rows.sort((A, B) => {
      if (k === 'edge') return d * (((A.c && !A.g.final) ? A.c.ae : -1) - ((B.c && !B.g.final) ? B.c.ae : -1));
      return d * (new Date(A.g.kickoff) - new Date(B.g.kickoff));
    });
    const live = W.games.filter((g) => !g.final).map((g) => (isSpread ? spreadCalc(g) : totalCalc(g))).filter(Boolean);
    const n5 = live.filter((c) => c.ae >= 5).length, n7 = live.filter((c) => c.ae >= 7).length;
    const nLines = W.games.filter((g) => g.line && (isSpread ? g.line.spread != null : g.line.total != null)).length;
    const what = isSpread ? 'DraftKings line' : 'DraftKings total';
    const chance = isSpread ? "Cover chance is for the model's side with pushes refunded" : "Hit chance is for the model's side of the total";
    const lead = nLines
      ? `<b>${n5} games</b> where the model is 5+ points off the ${what} (<b>${n7}</b> at 7+). ${chance}; break-even at -110 is ${BREAK_EVEN}%.`
      : `No ${what}s posted for week ${W.week} yet. The daily job pulls DraftKings each morning; model numbers are already below.`;
    const sortArr = (key) => (UI.wsort.k === key ? `<span class="arr">${UI.wsort.d > 0 ? '▲' : '▼'}</span>` : '');
    el.innerHTML = `
      <div class="card">
        <div class="cardhead">
          <div><h2>This week's games, week ${W.week}</h2></div>
          <span class="spacer"></span>
          <div class="seg" id="viewSeg"><button data-v="spread" class="${isSpread ? 'on' : ''}">Spreads</button><button data-v="total" class="${!isSpread ? 'on' : ''}">Totals</button></div>
          <div class="seg" id="edgeSeg"><button data-e="0" class="${UI.minEdge === 0 ? 'on' : ''}">All</button><button data-e="5" class="${UI.minEdge === 5 ? 'on' : ''}">5+ pts</button><button data-e="7" class="${UI.minEdge === 7 ? 'on' : ''}">7+ pts</button></div>
        </div>
        <p class="lead">${lead}</p>
        <div class="filters">
          <input id="wq" placeholder="Search a team" value="${esc(UI.q)}">
          <select id="wconf"><option value="">All conferences</option>${confs.map((c) => `<option ${c === UI.conf ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>
        </div>
        <div class="tblwrap"><table>
          <thead><tr>
            <th>Matchup</th><th class="sort" data-s="kickoff">Kickoff ${sortArr('kickoff')}</th>
            <th class="num">Line</th><th class="num">Model</th><th>Model side</th>
            <th class="num sort" data-s="edge">Edge ${sortArr('edge')}</th><th>${isSpread ? 'Cover chance' : 'Hit chance'}</th><th></th><th></th>
          </tr></thead>
          <tbody>${rows.map(({ g, c }) => weekRow(g, c, isSpread)).join('') || `<tr><td colspan="9" class="empty">No games match.</td></tr>`}</tbody>
        </table></div>
        <p class="note">Kickoff times are Central. ${weekFootnote()}</p>
      </div>`;
    el.querySelectorAll('#viewSeg button').forEach((b) => (b.onclick = () => { UI.view = b.dataset.v; renderWeek(); }));
    el.querySelectorAll('#edgeSeg button').forEach((b) => (b.onclick = () => { UI.minEdge = +b.dataset.e; renderWeek(); }));
    el.querySelectorAll('th.sort').forEach((th) => (th.onclick = () => {
      const k2 = th.dataset.s; UI.wsort = { k: k2, d: UI.wsort.k === k2 ? -UI.wsort.d : (k2 === 'edge' ? -1 : 1) }; renderWeek();
    }));
    const wq = $('#wq'); wq.oninput = () => { UI.q = wq.value; renderWeek(); const n = $('#wq'); n.focus(); n.setSelectionRange(n.value.length, n.value.length); };
    $('#wconf').onchange = (e) => { UI.conf = e.target.value; renderWeek(); };
    el.querySelectorAll('[data-open]').forEach((b) => (b.onclick = () => { UI.gameId = b.dataset.open; UI.custom = null; setTab('detail'); }));
  }
  function weekRow(g, c, isSpread) {
    const L = g.line;
    let line = '<span class="dim">–</span>', model, side = '<span class="dim">–</span>', edge = '<span class="dim">–</span>', ch = '<span class="dim">–</span>', flag = '';
    const src = L ? `<span class="tag dk">${L.source === 'DK' ? 'DK' : 'ESPN'}</span>` : '';
    if (isSpread) {
      const m = adjMargin(g);
      model = `${esc(abbr(m >= 0 ? g.home : g.away))} ${fmtLine(-Math.abs(m))}`;
      if (L && L.spread != null) line = `${esc(abbr(L.spread <= 0 ? g.home : g.away))} ${fmtLine(-Math.abs(L.spread))}`;
      if (c && !g.final) {
        side = `<b>${esc(abbr(c.sideTeam))} ${fmtLine(c.sideLine)}</b>`;
        edge = num(c.ae, 1);
        ch = `${pct(c.p)}${c.push >= 0.01 ? ` <span class="dim">push ${pct(c.push)}</span>` : ''}`;
        flag = flagHtml(c.ae);
      }
    } else {
      const t = adjTotal(g);
      model = num(t, 1);
      if (L && L.total != null) line = num(L.total, 1);
      if (c && !g.final) {
        side = `<b>${c.over ? 'Over' : 'Under'} ${trim(L.total)}</b>`;
        edge = num(c.ae, 1);
        ch = pct(c.p);
        flag = flagHtml(c.ae);
      }
    }
    if (g.final) {
      side = `<span class="final">final ${g.score.away}-${g.score.home}</span>`;
    }
    const cls = !g.final && c ? flagOf(c.ae) : '';
    return `<tr class="${cls}"><td class="mcell">${matchup(g)}</td><td class="mono">${ctTime(g.kickoff, g.tbd)}</td>
      <td class="num">${line} ${L ? src : ''}</td><td class="num">${model}</td><td>${side}</td>
      <td class="num">${edge}</td><td class="mono">${ch}</td><td>${flag}</td>
      <td><button class="open" data-open="${g.id}">Open</button></td></tr>`;
  }
  function weekFootnote() {
    const s = S.site;
    const bits = [];
    bits.push(s.dk_lines ? `DraftKings lines on ${s.dk_lines} of ${s.games_this_week} games.` : 'DraftKings lines not loaded yet.');
    bits.push(`Weather: ${s.weather_loaded} outdoor forecasts loaded.`);
    if (s.unmatched_odds && s.unmatched_odds.length) bits.push(`${s.unmatched_odds.length} DraftKings games could not be matched to the schedule.`);
    bits.push('QB CHANGE means the last start was not the season\'s main QB.');
    return bits.join(' ');
  }
  function confOf(id) { const t = S.ratingsById[id]; return t ? t.conf : ''; }

  // ---------- GAME DETAIL ----------
  function currentGame() {
    if (UI.custom) return customGame(UI.custom.home, UI.custom.away, UI.custom.neutral);
    let g = S.week.games.find((x) => String(x.id) === String(UI.gameId));
    if (!g) {
      const open = S.week.games.filter((x) => !x.final);
      g = open.map((x) => ({ x, c: spreadCalc(x) })).sort((a, b) => ((b.c ? b.c.ae : -1) - (a.c ? a.c.ae : -1)))[0];
      g = g ? g.x : S.week.games[0];
      UI.gameId = g ? g.id : null;
    }
    return g;
  }
  function renderDetail() {
    const el = $('#tab-detail');
    const g = currentGame();
    if (!g) { el.innerHTML = '<div class="card empty">No games this week.</div>'; return; }
    const teams = S.ratings.teams.slice().sort((a, b) => a.school.localeCompare(b.school));
    const opt = (sel) => teams.map((t) => `<option value="${t.id}" ${t.id === sel ? 'selected' : ''}>${esc(t.school)}</option>`).join('');
    const sc = spreadCalc(g), tc = totalCalc(g);
    const m = adjMargin(g);
    const favM = m >= 0 ? g.home : g.away;
    const tp = totalParts(g);
    const L = g.line;
    const gameList = S.week.games.map((x) => `<option value="${x.id}" ${String(x.id) === String(g.id) ? 'selected' : ''}>${esc(x.away.school)} ${x.neutral ? 'vs' : '@'} ${esc(x.home.school)}${x.final ? ' (final)' : ''}</option>`).join('');
    el.innerHTML = `
      <div class="card">
        <div class="gd-head"><h2>${esc(abbr(g.away))} ${g.neutral ? 'vs' : '@'} ${esc(abbr(g.home))}</h2>
          <span class="when">${g.custom ? 'Custom matchup · stats-only model (no FPI)' : esc(ctTime(g.kickoff, g.tbd)) + (g.venue ? ' · ' + esc(g.venue) : '')}</span></div>
        <div class="row"><select id="gsel"><option value="">Pick a game this week…</option>${gameList}</select></div>
        <div class="pick">
          <div><label>Away</label><select id="selA">${opt(g.away.id)}</select></div>
          <div class="atsign">@</div>
          <div><label>Home</label><select id="selH">${opt(g.home.id)}</select></div>
        </div>
        <label class="check"><input type="checkbox" id="neut" ${g.neutral ? 'checked' : ''}> Neutral site</label>
        <div class="stats">
          <div class="stat"><div class="k">Model spread</div><div class="v">${esc(abbr(favM))} ${fmtLine(-Math.abs(m))}</div>
            <div class="s">${L && L.spread != null ? `Line: ${esc(abbr(L.spread <= 0 ? g.home : g.away))} ${fmtLine(-Math.abs(L.spread))} (${L.source === 'DK' ? 'DraftKings' : 'ESPN'})` : 'No line posted yet'}</div></div>
          <div class="stat"><div class="k">Model side</div><div class="v">${sc ? esc(abbr(sc.sideTeam)) + ' ' + fmtLine(sc.sideLine) : '–'}</div>
            <div class="s">The bet the model leans toward at the posted line</div></div>
          <div class="stat"><div class="k">Model edge</div><div class="v ${sc ? (sc.ae >= 7 ? 'g7' : sc.ae >= 5 ? 'g5' : '') : ''}">${sc ? num(sc.ae, 1) + ' pts' : '–'}</div>
            <div class="s">How far the model is from the line · flags at 5+ and 7+</div></div>
          <div class="stat"><div class="k">${sc ? esc(abbr(sc.sideTeam)) + ' ' + fmtLine(sc.sideLine) + ' covers' : 'Cover chance'}</div><div class="v">${sc ? pct(sc.p) : '–'}</div>
            <div class="s">Break-even at -110 is ${BREAK_EVEN}%</div></div>
          <div class="stat"><div class="k">Total</div><div class="v">${tc ? (tc.over ? 'Over ' : 'Under ') + trim(L.total) : num(tp.total, 1)}</div>
            <div class="s">Model ${num(tp.total, 1)}${tc ? ' · hits ' + pct(tc.p) + ' · edge ' + num(tc.ae, 1) : ''}</div></div>
        </div>
        ${sc ? ladderHtml(g, sc) : ''}
      </div>
      <div class="grid2">
        <div class="card">${breakdownHtml(g)}</div>
        <div class="card">${matchupStatsHtml(g)}</div>
      </div>
      <div class="grid2" style="margin-top:16px">
        <div class="card" id="injCard">${injuriesHtml(g)}</div>
        <div class="card" id="wxCard">${weatherHtml(g, tp)}</div>
      </div>`;
    $('#gsel').onchange = (e) => { if (e.target.value) { UI.gameId = e.target.value; UI.custom = null; renderDetail(); } };
    const onPick = () => {
      const h = +$('#selH').value, a = +$('#selA').value, n = $('#neut').checked;
      const real = S.week.games.find((x) => x.home.id === h && x.away.id === a);
      if (real && n === real.neutral) { UI.gameId = real.id; UI.custom = null; } else { UI.custom = { home: h, away: a, neutral: n }; }
      renderDetail();
    };
    $('#selH').onchange = onPick; $('#selA').onchange = onPick; $('#neut').onchange = onPick;
    bindInjuries(g); bindWeather(g);
  }

  function ladderHtml(g, sc) {
    const s = sc.sideLine;
    const base = sc.p * (1 - (Number.isInteger(s) ? pushRate(Math.abs(s)) : 0));   // unconditional chance of a win at the posted line
    const mass = (m) => pushRate(Math.abs(m));       // chance the side's final margin is exactly m
    const xs = [];
    for (let k = 4; k >= -4; k--) xs.push(s + k * 0.5);
    const cells = xs.map((x) => {
      // the side covers when its margin m > -x; moving the line adds or removes the integer margins in between
      let p = base;
      if (x > s) { for (let m = Math.floor(-s); m > -x; m--) { if (m <= -s) p += mass(m); } }
      if (x < s) { for (let m = Math.floor(-x); m > -s; m--) { if (m <= -x) p -= mass(m); } }
      const push = Number.isInteger(x) ? mass(-x) : 0;
      p = Math.max(0.01, Math.min(0.99, p / (1 - push)));   // shown as win chance when the bet is not refunded
      const cls = x === s ? 'posted' : (Number.isInteger(x) && KEY_LINES.has(Math.abs(x)) ? 'key' : '');
      return `<div class="rung ${cls}"><div class="l">${esc(abbr(sc.sideTeam))} ${fmtLine(x)}</div><div class="p">${pct(p)}</div><div class="pu">${push >= 0.01 ? 'push ' + pct(push) : ''}</div></div>`;
    }).join('');
    const p3 = pct(S.model.key_margins['3'] || 0), p7 = pct(S.model.key_margins['7'] || 0);
    return `<div class="card" style="margin:0;background:var(--panel-2)"><h3>Key number ladder</h3>
      <p class="lead" style="margin-top:0">Chance ${esc(abbr(sc.sideTeam))} covers at each line you might find, best on the left. The filled box is the posted line and outlined boxes are key numbers. Chances leave out pushes (refunded bets), which are listed underneath. ${p3} of FBS games end on exactly 3 and ${p7} on exactly 7, so crossing those numbers is worth the most.</p>
      <div class="ladder">${cells}</div></div>`;
  }

  function breakdownHtml(g) {
    const rows = (g.model.comps || []).map((c) => `<tr><td>${esc(c.label)}</td><td class="${c.pts > 0.05 ? 'pos' : c.pts < -0.05 ? 'neg' : ''}">${sgn(c.pts, 2)}</td></tr>`).join('');
    const o = overrides(g);
    const qb = S.model.spread.qb_change_points;
    const extra = (o.qbHome ? `<tr><td>${esc(abbr(g.home))} QB out (override)</td><td class="neg">−${qb.toFixed(2)}</td></tr>` : '') +
      (o.qbAway ? `<tr><td>${esc(abbr(g.away))} QB out (override)</td><td class="pos">+${qb.toFixed(2)}</td></tr>` : '') +
      (o.useExtras && o.extras.length ? `<tr><td>Other injuries (rough)</td><td>${sgn(o.extras.reduce((a, e) => a + (e.side === 'home' ? -e.val : e.val), 0), 2)}</td></tr>` : '');
    const m = adjMargin(g);
    const s = g.situ || {};
    const ctx = [];
    if (s.travel_away_km != null) ctx.push(`${esc(g.away.school)} travel ${Math.round(s.travel_away_km * 0.621)} mi`);
    if (s.rest_home != null && s.rest_away != null) ctx.push(`rest ${num(s.rest_away, 0)} vs ${num(s.rest_home, 0)} days`);
    if (s.elev_m != null && s.elev_m > 1200) ctx.push(`elevation ${Math.round(s.elev_m * 3.281).toLocaleString()} ft`);
    return `<h3>Why the model says ${esc(abbr(m >= 0 ? g.home : g.away))} ${fmtLine(-Math.abs(m))}</h3>
      <p class="lead" style="margin-top:0">Each line is points toward ${esc(g.home.school)} (positive) or ${esc(g.away.school)} (negative). ${g.model.uses_fpi ? 'ESPN FPI already includes home field, so the home-field term and the baseline mostly offset.' : 'No FPI for this game, so the stats-only version of the model is used.'}</p>
      <table class="kv">${rows}${extra}<tr class="sum"><td>Model margin (${esc(g.home.school)})</td><td>${sgn(m, 1)}</td></tr></table>
      <p class="note">${ctx.length ? 'Context: ' + ctx.join(' · ') + '. ' : ''}Travel, rest, bye weeks, kickoff time and altitude were tested and did not beat the closing line, so they are shown for context only (see Method).</p>`;
  }

  function matchupStatsHtml(g) {
    const h = S.ratingsById[g.home.id], a = S.ratingsById[g.away.id];
    if (!h || !a) return '<h3>Team comparison</h3><p class="empty">Ratings not available.</p>';
    const row = (label, k, d = 1, better = 1, fmt) => {
      const av = a[k], hv = h[k];
      const f = fmt || ((v) => num(v, d));
      const aw = av != null && hv != null && (better > 0 ? av > hv : av < hv);
      const hw = av != null && hv != null && (better > 0 ? hv > av : hv < av);
      return `<tr><td>${label}</td><td class="num ${aw ? 'good' : ''}">${f(av)}</td><td class="num ${hw ? 'good' : ''}">${f(hv)}</td></tr>`;
    };
    return `<h3>Team comparison</h3>
      <div class="tblwrap"><table><thead><tr><th></th><th class="num">${esc(abbr(g.away))}</th><th class="num">${esc(abbr(g.home))}</th></tr></thead><tbody>
      <tr><td>Record</td><td class="num">${a.record}</td><td class="num">${h.record}</td></tr>
      ${row('Power (pts vs avg FBS)', 'power', 1, 1, (v) => sgn(v, 1))}
      ${row('Off success rate', 'off_sr', 1, 1, (v) => num(v, 1) + '%')}
      ${row('Def success rate', 'def_sr', 1, -1, (v) => num(v, 1) + '%')}
      ${row('Off EPA / play', 'off_epa', 3, 1, (v) => sgn(v, 3))}
      ${row('Def EPA / play', 'def_epa', 3, -1, (v) => sgn(v, 3))}
      ${row('Off yards / play', 'off_ypp', 2, 1)}
      ${row('Def yards / play', 'def_ypp', 2, -1)}
      ${row('Points per game', 'ppg', 1, 1)}
      ${row('Points allowed', 'opp_ppg', 1, -1)}
      <tr><td>Main QB</td><td class="num">${esc(a.qb || '–')}</td><td class="num">${esc(h.qb || '–')}</td></tr>
      </tbody></table></div>
      <p class="note">Rates are opponent-adjusted and exclude garbage time. Points per game are raw.</p>`;
  }

  function injuriesHtml(g) {
    const o = overrides(g);
    const qbPts = S.model.spread.qb_change_points;
    const side = (t, key) => {
      const q = t.qb || {};
      const changed = q.changed ? `<span class="tag red">last start: ${esc(q.last)}</span>` : '';
      return `<div class="sec-label">${esc(t.school)}</div>
        <div class="row"><select data-qb="${key}">
          <option value="0" ${!o[key] ? 'selected' : ''}>${esc(q.primary || 'Starter')} starts (expected)</option>
          <option value="1" ${o[key] ? 'selected' : ''}>Starter out, backup starts</option>
        </select></div>${changed}`;
    };
    const ex = o.extras.map((e, i) => `<tr><td>${esc(e.side === 'home' ? abbr(g.home) : abbr(g.away))} · ${esc(e.pos)} · ${esc(e.status)}${e.note ? ' · ' + esc(e.note) : ''}</td>
      <td>${(e.side === 'home' ? -e.val : e.val) >= 0 ? '+' : '−'}${Math.abs(e.val).toFixed(2)} <button class="open" data-del="${i}">remove</button></td></tr>`).join('');
    return `<h3>Injuries &amp; QB <span class="tag">manual</span></h3>
      <p class="lead" style="margin-top:0">College injury reports are not reliable enough to load automatically, so this is yours to set. A QB change is worth <b>${qbPts.toFixed(1)} points</b> on the spread and ${Math.abs(S.model.total.qb_change_points).toFixed(1)} off the total (fitted on 2019-2025). Use it when the news is fresh: the backtest shows the market already prices a QB change once it has happened in a game.</p>
      ${side(g.away, 'qbAway')}${side(g.home, 'qbHome')}
      <div class="sec-label">Other starters <span class="tag">rough values, not backtested</span></div>
      <label class="check" style="margin:6px 0"><input type="checkbox" id="useEx" ${o.useExtras ? 'checked' : ''}> Count other injured starters</label>
      <table class="inj">${ex || '<tr><td class="dim">No manual entries</td><td></td></tr>'}</table>
      <div class="row">
        <select id="exSide"><option value="away">${esc(g.away.school)}</option><option value="home">${esc(g.home.school)}</option></select>
        <select id="exPos">${EXTRA_POS.filter((p) => p[1] > 0).map((p) => `<option value="${p[1]}">${p[0]}</option>`).join('')}</select>
        <select id="exSt"><option>Out</option><option>Doubtful</option><option>Questionable</option></select>
      </div>
      <div class="row"><input id="exNote" placeholder="Note, e.g. player name"><button class="btn-acc" id="exAdd">Add</button></div>
      <p class="note">Saved in this browser only.</p>`;
  }
  function bindInjuries(g) {
    const card = $('#injCard'); if (!card) return;
    card.querySelectorAll('[data-qb]').forEach((s) => (s.onchange = () => { const o = overrides(g); o[s.dataset.qb] = +s.value; saveOverrides(g, o); renderDetail(); }));
    $('#useEx').onchange = (e) => { const o = overrides(g); o.useExtras = e.target.checked; saveOverrides(g, o); renderDetail(); };
    $('#exAdd').onclick = () => {
      const o = overrides(g);
      const sel = $('#exPos'); const base = +sel.value; const st = $('#exSt').value;
      o.extras.push({ side: $('#exSide').value, pos: sel.options[sel.selectedIndex].text, status: st, val: +(base * STATUS_MULT[st]).toFixed(2), note: $('#exNote').value });
      o.useExtras = true; saveOverrides(g, o); renderDetail();
    };
    card.querySelectorAll('[data-del]').forEach((b) => (b.onclick = () => { const o = overrides(g); o.extras.splice(+b.dataset.del, 1); saveOverrides(g, o); renderDetail(); }));
  }

  function weatherHtml(g, tp) {
    const wx = tp.wx;
    const W = S.model.total.weather;
    const status = g.dome ? 'indoor' : (g.weather ? 'forecast' : 'no forecast yet');
    const rows = tp.parts.map((p) => `<tr><td>${esc(p[0])}</td><td>${p[0].startsWith('Base') ? num(p[1], 1) : sgn(p[1], 1)}</td></tr>`).join('');
    const note = W.fitted ? `Weather values are fitted on ${W.n_games} outdoor FBS games.` : 'Weather values are starting estimates from published research and are not yet fitted on this data; the first full retrain backfills weather history and refits them.';
    return `<h3>Game-day weather <span class="tag">${status}</span></h3>
      <div class="pick" style="grid-template-columns:1fr 1fr">
        <div><label>Venue</label><select id="wxVenue"><option value="outdoor" ${wx.venue !== 'dome' ? 'selected' : ''}>Outdoor</option><option value="dome" ${wx.venue === 'dome' ? 'selected' : ''}>Dome / roof closed</option></select></div>
        <div><label>Precipitation</label><select id="wxPrec"><option value="none" ${wx.precip === 'none' ? 'selected' : ''}>None</option><option value="rain" ${wx.precip === 'rain' ? 'selected' : ''}>Rain</option><option value="snow" ${wx.precip === 'snow' ? 'selected' : ''}>Snow</option></select></div>
      </div>
      <div class="lbl"><span>Wind</span><b id="wxWv">${wx.wind} mph</b></div>
      <input class="slider" type="range" min="0" max="40" value="${wx.wind}" id="wxWind">
      <div class="lbl"><span>Temperature</span><b id="wxTv">${wx.temp}°F</b></div>
      <input class="slider" type="range" min="-5" max="100" value="${wx.temp}" id="wxTemp">
      <table class="kv" style="margin-top:12px">${rows}<tr class="sum"><td>Projected total</td><td>${num(tp.total, 1)}</td></tr></table>
      <p class="note">${g.dome ? 'Indoor game, weather does not apply. ' : ''}Weather moves the total only. ${note}</p>
      <button class="btn-ghost" id="wxReset">Reset to forecast</button>`;
  }
  function bindWeather(g) {
    const set = () => {
      const o = overrides(g);
      o.wx = { venue: $('#wxVenue').value, precip: $('#wxPrec').value, wind: +$('#wxWind').value, temp: +$('#wxTemp').value };
      saveOverrides(g, o); renderDetail();
    };
    $('#wxVenue').onchange = set; $('#wxPrec').onchange = set;
    $('#wxWind').oninput = (e) => { $('#wxWv').textContent = e.target.value + ' mph'; };
    $('#wxTemp').oninput = (e) => { $('#wxTv').textContent = e.target.value + '°F'; };
    $('#wxWind').onchange = set; $('#wxTemp').onchange = set;
    $('#wxReset').onclick = () => { const o = overrides(g); o.wx = null; saveOverrides(g, o); renderDetail(); };
  }

  // ---------- RESULTS ----------
  function renderResults() {
    const el = $('#tab-results');
    const R = S.results.games;
    const weeks = [...new Set(R.map((r) => r.wk))].sort((a, b) => a - b);
    if (!weeks.length) { el.innerHTML = '<div class="card empty">No completed games yet this season.</div>'; return; }
    if (UI.resWeek == null || !weeks.includes(UI.resWeek)) UI.resWeek = weeks[weeks.length - 1];
    const wk = UI.resWeek;
    const rows = R.filter((r) => r.wk === wk);
    const rec = (xs, k = 'result') => { const w = xs.filter((x) => x[k] === true).length, l = xs.filter((x) => x[k] === false).length; return { w, l, s: `${w}-${l}`, p: w + l ? w / (w + l) : null }; };
    const all = rec(R), r5 = rec(R.filter((x) => Math.abs(x.edge) >= 5)), r7 = rec(R.filter((x) => Math.abs(x.edge) >= 7));
    const ou = rec(R.filter((x) => x.total_result != null), 'total_result');
    const ou5 = rec(R.filter((x) => x.total_result != null && Math.abs(x.total_edge) >= 5), 'total_result');
    const wr = rec(rows), w5 = rec(rows.filter((x) => Math.abs(x.edge) >= 5));
    const byWeek = weeks.map((w) => { const xs = R.filter((r) => r.wk === w); const a = rec(xs), f = rec(xs.filter((x) => Math.abs(x.edge) >= 5)), t = rec(xs.filter((x) => x.total_result != null), 'total_result'); return { w, a, f, t }; });
    const posted = rows.filter((r) => r.posted);
    el.innerHTML = `
      <div class="card">
        <div class="cardhead"><div><h2>${S.site.season} season so far</h2></div></div>
        <div class="stats">
          <div class="stat"><div class="k">Spreads, all games</div><div class="v">${all.s}</div><div class="s">${pct(all.p, 1)} · break-even ${BREAK_EVEN}%</div></div>
          <div class="stat"><div class="k">5+ edge</div><div class="v g5">${r5.s}</div><div class="s">${pct(r5.p, 1)}</div></div>
          <div class="stat"><div class="k">7+ edge</div><div class="v g7">${r7.s}</div><div class="s">${pct(r7.p, 1)}</div></div>
          <div class="stat"><div class="k">Totals, all games</div><div class="v">${ou.s}</div><div class="s">${pct(ou.p, 1)}</div></div>
          <div class="stat"><div class="k">Totals 5+ edge</div><div class="v">${ou5.s}</div><div class="s">${pct(ou5.p, 1)}</div></div>
        </div>
        <p class="note">Graded against the DraftKings game-day line once the daily job has tracked a game; before that, against ESPN's closing line. Every pick uses only information from before kickoff, and the model was trained on 2019-2025 only, so ${S.site.season} is fully out of sample. Pushes are not counted.</p>
        <div class="tblwrap"><table><thead><tr><th>Week</th><th class="num">Spreads</th><th class="num">5+ edge</th><th class="num">Totals</th></tr></thead><tbody>
        ${byWeek.map((b) => `<tr class="click" data-wk="${b.w}"><td>Week ${b.w}${b.w === wk ? ' <span class="tag">shown</span>' : ''}</td><td class="num">${b.a.s} <span class="dim">${pct(b.a.p)}</span></td><td class="num">${b.f.s}</td><td class="num">${b.t.s}</td></tr>`).join('')}
        </tbody></table></div>
      </div>
      <div class="card">
        <div class="cardhead"><div><h2>Week ${wk}, model vs actual</h2></div><span class="spacer"></span>
          <select class="inp" id="rw">${weeks.map((w) => `<option value="${w}" ${w === wk ? 'selected' : ''}>Week ${w}</option>`).join('')}</select></div>
        <p class="lead">Spreads ${wr.s} (${pct(wr.p)}), 5+ edge ${w5.s}. Line shown is the one the pick was graded against.</p>
        <div class="tblwrap"><table><thead><tr><th>Matchup</th><th class="num">Line</th><th>Source</th><th class="num">Model</th><th>Model side</th><th class="num">Edge</th><th class="num">Final</th><th>Result</th><th class="num">Total</th><th class="num">Model</th><th>O/U</th></tr></thead><tbody>
        ${rows.sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge)).map(resRow).join('')}
        </tbody></table></div>
      </div>
      <div class="card">
        <h2>Week ${wk}, line movement</h2>
        ${posted.length ? `<p class="lead">Model side when the line was first posted vs the game-day line. "Toward model" means the market moved to agree with the model, which is the best sign a model is finding real value.</p>
        <div class="tblwrap"><table><thead><tr><th>Matchup</th><th>Model side when posted</th><th>Game-day line</th><th class="num">Edge</th><th class="num">Move</th><th>Direction</th></tr></thead><tbody>
        ${posted.map(moveRow).join('')}</tbody></table></div>` : `<p class="lead">Line tracking starts with the first daily run that has DraftKings lines. Each game gets a snapshot when its line first appears and another on game day.</p>`}
      </div>
      <div class="grid2">
        <div class="card">${calibHtml()}</div>
        <div class="card">${marginsHtml()}</div>
      </div>`;
    $('#rw').onchange = (e) => { UI.resWeek = +e.target.value; renderResults(); };
    el.querySelectorAll('[data-wk]').forEach((t) => (t.onclick = () => { UI.resWeek = +t.dataset.wk; renderResults(); }));
  }
  function resRow(r) {
    const homeSide = r.edge >= 0;
    const sideAbbr = homeSide ? r.home_abbr : r.away_abbr;
    const sideLine = homeSide ? r.line : -r.line;
    const favAbbr = r.line <= 0 ? r.home_abbr : r.away_abbr;
    const modelFav = r.model_margin >= 0 ? r.home_abbr : r.away_abbr;
    const res = r.push ? '<span class="pill push">Push</span>' : r.result ? '<span class="pill win">Win</span>' : '<span class="pill loss">Loss</span>';
    const ae = Math.abs(r.edge);
    const ou = r.total_result == null ? '<span class="dim">–</span>' : `${r.total_edge >= 0 ? 'O' : 'U'} ${r.total_result ? '<span class="pill win">W</span>' : '<span class="pill loss">L</span>'}`;
    return `<tr class="${flagOf(ae)}"><td><span class="mt">${esc(r.away)}<span class="at">${r.neutral ? 'vs' : '@'}</span>${esc(r.home)}</span></td>
      <td class="num">${esc(favAbbr)} ${fmtLine(-Math.abs(r.line))}</td><td><span class="tag dk">${esc(r.line_src)}</span></td>
      <td class="num">${esc(modelFav)} ${fmtLine(-Math.abs(r.model_margin))}</td><td><b>${esc(sideAbbr)} ${fmtLine(sideLine)}</b></td>
      <td class="num">${num(ae, 1)}</td><td class="num">${r.score_away}-${r.score_home}</td><td>${res}</td>
      <td class="num">${r.total_line != null ? trim(r.total_line) : '–'}</td><td class="num">${r.model_total != null ? num(r.model_total, 1) : '–'}</td><td>${ou}</td></tr>`;
  }
  function moveRow(r) {
    const p = r.posted;
    const homeSide = (p.model_margin + p.spread) >= 0;
    const side = homeSide ? r.home_abbr : r.away_abbr;
    const posted = homeSide ? p.spread : -p.spread;
    const day = homeSide ? r.line : -r.line;
    const mv = posted - day;   // positive: the model side now gets fewer points, market moved toward the model
    const dir = Math.abs(mv) < 0.01 ? '<span class="pill none">No move</span>' : mv > 0 ? '<span class="pill toward">Toward model</span>' : '<span class="pill away">Away from model</span>';
    const posTs = new Date(p.ts).toLocaleDateString('en-US', { timeZone: 'America/Chicago', weekday: 'short' });
    return `<tr><td><span class="mt">${esc(r.away_abbr)}<span class="at">@</span>${esc(r.home_abbr)}</span></td>
      <td><b>${esc(side)} ${fmtLine(posted)}</b> <span class="dim">${posTs} · DK</span></td><td>${esc(side)} ${fmtLine(day)} <span class="dim">${esc(r.line_src)}</span></td>
      <td class="num">${num(Math.abs(p.model_margin + p.spread), 1)}</td><td class="num">${sgn(mv, 2)}</td><td>${dir}</td></tr>`;
  }
  function calibHtml() {
    const c = S.backtest.calibration.spread;
    const seasons = S.backtest.coverage.spread_seasons;
    return `<h3>Is the cover chance honest?</h3>
      <p class="lead" style="margin-top:0">Every ${seasons[0]}-${seasons[seasons.length - 1]} FBS game, grouped by the cover chance the page would have shown for the model's side, next to how often that side actually covered. Each season was scored by a version fitted on the other seasons.</p>
      <div class="tblwrap"><table><thead><tr><th>Page would show</th><th class="num">Games</th><th class="num">Average shown</th><th class="num">Actually covered</th></tr></thead><tbody>
      ${c.map((b) => `<tr><td>${esc(b.bucket)}</td><td class="num">${b.n}</td><td class="num">${pct(b.shown, 1)}</td><td class="num">${pct(b.actual, 1)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="note">The shown chance stays close to 50% on purpose. Bigger model edges only win slightly more often, and an honest page should say so.</p>`;
  }
  function marginsHtml() {
    const km = S.model.key_margins;
    const max = Math.max(...Object.values(km));
    const hot = new Set([3, 7, 10, 14, 17, 21]);
    return `<h3>How often games end on each margin</h3>
      <p class="lead" style="margin-top:0">Share of 2019-2025 FBS games decided by exactly this many points. College margins are spread out more than the NFL, but 3, 7, 10, 14, 17 and 21 still stand out, which is why some half points are worth more than others.</p>
      <div class="bars">${Object.entries(km).slice(0, 24).map(([k, v]) => `<span>${k}</span><div class="bar ${hot.has(+k) ? 'hot' : ''}" style="width:${(v / max) * 100}%"></div><span class="r">${(v * 100).toFixed(1)}%</span>`).join('')}</div>`;
  }

  // ---------- RATINGS ----------
  function renderRatings() {
    const el = $('#tab-ratings');
    const R = S.ratings;
    const confs = [...new Set(R.teams.map((t) => t.conf).filter(Boolean))].sort();
    let teams = R.teams.slice();
    if (UI.rconf) teams = teams.filter((t) => t.conf === UI.rconf);
    if (UI.rq) teams = teams.filter((t) => t.school.toLowerCase().includes(UI.rq.toLowerCase()));
    const k = UI.rsort.k, d = UI.rsort.d;
    teams.sort((a, b) => { const x = a[k], y = b[k]; if (typeof x === 'string') return d * x.localeCompare(y); return d * ((x == null ? -1e9 : x) - (y == null ? -1e9 : y)); });
    const cols = [['school', 'Team', 0], ['conf', 'Conf', 0], ['gp', 'GP', 1], ['record', 'Rec', 0], ['power', 'Power', 1], ['off_sr', 'Off SR', 1], ['def_sr', 'Def SR', 1],
      ['off_epa', 'Off EPA/play', 1], ['def_epa', 'Def EPA/play', 1], ['off_ypp', 'Off YPP', 1], ['def_ypp', 'Def YPP', 1], ['ppg', 'PPG', 1], ['opp_ppg', 'Opp PPG', 1], ['qb', 'QB', 0]];
    const head = cols.map(([key, label, n]) => `<th class="sort ${n ? 'num' : ''}" data-k="${key}">${label}${UI.rsort.k === key ? ` <span class="arr">${UI.rsort.d > 0 ? '▲' : '▼'}</span>` : ''}</th>`).join('');
    el.innerHTML = `
      <div class="card">
        <h2>Team ratings</h2>
        <p class="lead">Opponent-adjusted, garbage time removed, built from every FBS play this season and shrunk toward a preseason starting point (last season's rating plus returning production, QB continuity and transfers). Power = points better than an average FBS team on a neutral field from the stats alone; the game spreads also blend in ESPN FPI. Ratings are what the model knew before week ${R.week}. Click a column to sort.</p>
        <div class="filters"><input id="rq" placeholder="Search a team" value="${esc(UI.rq)}"><select id="rconf"><option value="">All conferences</option>${confs.map((c) => `<option ${c === UI.rconf ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></div>
        <div class="tblwrap"><table><thead><tr><th class="num">#</th>${head}</tr></thead><tbody>
        ${teams.map((t, i) => `<tr><td class="num dim">${i + 1}</td><td><b>${esc(t.school)}</b></td><td class="dim">${esc(t.conf)}</td><td class="num">${t.gp}</td><td class="num">${t.record}</td>
          <td class="num"><b>${sgn(t.power, 1)}</b></td><td class="num">${num(t.off_sr, 1)}%</td><td class="num">${num(t.def_sr, 1)}%</td>
          <td class="num">${sgn(t.off_epa, 3)}</td><td class="num">${sgn(t.def_epa, 3)}</td><td class="num">${num(t.off_ypp, 2)}</td><td class="num">${num(t.def_ypp, 2)}</td>
          <td class="num">${num(t.ppg, 1)}</td><td class="num">${num(t.opp_ppg, 1)}</td><td>${esc(t.qb || '–')}${t.qb_changed ? ' <span class="tag red">last: ' + esc(t.qb_last) + '</span>' : ''}</td></tr>`).join('')}
        </tbody></table></div>
      </div>`;
    el.querySelectorAll('th.sort').forEach((th) => (th.onclick = () => { const key = th.dataset.k; UI.rsort = { k: key, d: UI.rsort.k === key ? -UI.rsort.d : (['school', 'conf', 'qb', 'def_sr', 'def_epa', 'def_ypp', 'opp_ppg'].includes(key) ? 1 : -1) }; renderRatings(); }));
    const rq = $('#rq'); rq.oninput = () => { UI.rq = rq.value; renderRatings(); const n = $('#rq'); n.focus(); n.setSelectionRange(n.value.length, n.value.length); };
    $('#rconf').onchange = (e) => { UI.rconf = e.target.value; renderRatings(); };
  }

  // ---------- METHOD & TESTING ----------
  function renderMethod() {
    const el = $('#tab-method');
    const B = S.backtest, M = S.model, T = S.tests;
    const r = (x) => (x ? `${x.w}-${x.l} <span class="dim">(${pct(x.pct, 1)})</span>` : '–');
    const seasons = Object.keys(B.spread.by_season);
    const tseasons = Object.keys(B.total.by_season);
    const co = M.spread.coef;
    const fnames = M.spread.features.map((f) => ({ x_sr: 'success_rate_edge', x_ypp: 'yards_per_play_edge', x_fpi: 'espn_fpi_margin', loc: 'home_field', fluck_diff: 'fumble_luck' }[f] || f));
    const formula = ['2', '1', '0'].map((b) => {
      const c = co[b];
      const lab = { 0: 'Weeks 1-3', 1: 'Weeks 4-6', 2: 'Weeks 7+ and bowls' }[b];
      return `${lab}:\n  margin = ${c[c.length - 1].toFixed(2)}` + fnames.map((f, i) => `\n         ${c[i] >= 0 ? '+' : '-'} ${Math.abs(c[i]).toFixed(c[i] > 5 || c[i] < -5 ? 1 : 3)} x ${f}`).join('');
    }).join('\n\n');
    const W = M.total.weather;
    const testRow = (x, kind) => {
      const base = T.final;
      const dm = x.mae - base.mae, da = x.ats - base.ats;
      const verdict = kind === 'removed' ? (dm > 0.01 ? '<span class="good">Kept: removing it hurts</span>' : '<span class="dim">Kept, small effect</span>')
        : (x.feature === 'qbchg_edge' ? '<span class="good">Manual override</span>' : (dm < -0.01 && da > 0.003 ? '<span class="good">Helps</span>' : '<span class="bad">Dropped: no gain</span>'));
      return `<tr><td>${esc(x.label)}</td><td class="num">${x.mae.toFixed(2)}</td><td class="num">${pct(x.ats, 1)}</td><td class="num">${pct(x.ats5, 1)} <span class="dim">${x.n5}</span></td><td class="num">${pct(x.ats7, 1)} <span class="dim">${x.n7}</span></td><td>${verdict}</td></tr>`;
    };
    el.innerHTML = `
      <div class="card prose">
        <h2>How the model works</h2>
        <p class="lead">The short version: rate every FBS team from play-by-play, adjust for who they played, blend with ESPN's FPI, and compare the resulting spread and total to DraftKings. Every number below comes from data the model could have had before kickoff.</p>
        <div class="steps">
          <div class="stepbox"><b>Collect every play</b>Full FBS play-by-play (about 160,000 plays a season) from the public SportsDataverse feed, which already carries expected points (EPA) and success flags for each play. Schedules, ESPN closing lines, ESPN FPI game projections and transfer data come from the same place. DraftKings lines come from The Odds API, weather from Open-Meteo.</div>
          <div class="stepbox"><b>Clean it</b>Drop garbage time (2nd quarter up 38+, 3rd up 28+, 4th up 22+), kneel-downs and special teams. What is left is real offense against real defense.</div>
          <div class="stepbox"><b>Adjust for opponents</b>For success rate, EPA per play, yards per play, points and pace, solve for every team's offense and defense at once with a weighted ridge regression, so a good game against a bad defense counts for less. Ratings are rebuilt before every week using only games already played.</div>
          <div class="stepbox"><b>Start from a sensible preseason number</b>Before a team has played much, its rating leans on a prior: last season's final rating, plus returning offensive production, whether the starting QB is back, and net transfer talent. These weights were fitted on how 2021-2025 seasons actually played out. As games pile up, the team's own results take over.</div>
          <div class="stepbox"><b>Turn ratings into a spread</b>A regression, fitted separately for weeks 1-3, 4-6 and 7+, combines the rating gaps with ESPN FPI, home field and regressed fumble luck to predict the home margin. When FPI is missing, a stats-only version takes over.</div>
          <div class="stepbox"><b>Turn ratings into a total</b>Adjusted scoring for both sides, combined pace, offense-vs-defense EPA, domes and this season's scoring environment. It predicts the median total, not the average, because totals are lopsided (a few shootouts drag the average up) and over/under bets are decided by the middle.</div>
          <div class="stepbox"><b>Compare to the market</b>Edge = model spread minus the DraftKings spread. Games at 5+ points are flagged light green and 7+ green. Cover chances come from the backtest, not from the size of the edge alone.</div>
        </div>
      </div>

      <div class="card">
        <h2>The spread formula</h2>
        <p class="lead">Fitted on every FBS vs FBS game from 2019-2025 with a closing line. Inputs are home minus away; margin is points for the home team. FPI already contains home field, so the home-field term and the baseline mostly cancel each other out; what is left is a small correction.</p>
        <div class="formula">${esc(formula)}</div>
        <p class="note">Success rate edge = (home offense SR + away defense SR allowed) − (away offense SR + home defense SR allowed), opponent-adjusted. Yards per play works the same way. Fumble luck = share of fumbles your games' teams recovered vs the expected 50%, which does not carry forward, so the model leans against teams that have been lucky.</p>
      </div>

      <div class="card">
        <h2>Backtest: spreads</h2>
        <p class="lead">Leave one season out: each season is predicted by a model fitted only on the other seasons, then graded against the closing line. ${B.coverage.spread_games.toLocaleString()} FBS vs FBS games, ${seasons[0]}-${seasons[seasons.length - 1]}. Average miss: model ${B.mae.spread_model.toFixed(2)} points, closing line ${B.mae.spread_line.toFixed(2)}. The market is more accurate on average, as it should be; the edge comes from the games where they disagree.</p>
        <div class="tblwrap"><table><thead><tr><th>Season</th><th class="num">All games</th><th class="num">5+ edge</th><th class="num">7+ edge</th></tr></thead><tbody>
          ${seasons.map((s) => `<tr><td>${s}</td><td class="num">${r(B.spread.by_season[s].all)}</td><td class="num">${r(B.spread.by_season[s]['5+'])}</td><td class="num">${r(B.spread.by_season[s]['7+'])}</td></tr>`).join('')}
          <tr><td><b>Total</b></td><td class="num"><b>${r(B.spread.all)}</b></td><td class="num"><b>${r(B.spread['5+'])}</b></td><td class="num"><b>${r(B.spread['7+'])}</b></td></tr>
        </tbody></table></div>
        <h3 style="margin-top:16px">By part of the season</h3>
        <div class="tblwrap"><table><thead><tr><th>Weeks</th><th class="num">All games</th><th class="num">5+ edge</th></tr></thead><tbody>
          ${Object.entries(B.spread.by_week).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${r(v.all)}</td><td class="num">${r(v['5+'])}</td></tr>`).join('')}
        </tbody></table></div>
        <p class="note">Year to year swings are large: 7+ edges went from ${pct(B.spread.by_season[seasons[1]]['7+'].pct)} to ${pct(B.spread.by_season[seasons[2]]['7+'].pct)} between ${seasons[1]} and ${seasons[2]}. A few hundred bets is not enough to separate a small real edge from luck, so treat the flags as a short list to look at, not an automatic bet.</p>
      </div>

      <div class="card">
        <h2>Backtest: totals</h2>
        <p class="lead">Closing totals are in the data from 2023, so the totals backtest covers ${tseasons[0]}-${tseasons[tseasons.length - 1]} (${B.coverage.total_games.toLocaleString()} games). Average miss: model ${B.mae.total_model.toFixed(2)}, closing total ${B.mae.total_line.toFixed(2)}.</p>
        <div class="tblwrap"><table><thead><tr><th>Season</th><th class="num">All games</th><th class="num">5+ edge</th><th class="num">7+ edge</th></tr></thead><tbody>
          ${tseasons.map((s) => `<tr><td>${s}</td><td class="num">${r(B.total.by_season[s].all)}</td><td class="num">${r(B.total.by_season[s]['5+'])}</td><td class="num">${r(B.total.by_season[s]['7+'])}</td></tr>`).join('')}
          <tr><td><b>Total</b></td><td class="num"><b>${r(B.total.all)}</b></td><td class="num"><b>${r(B.total['5+'])}</b></td><td class="num"><b>${r(B.total['7+'])}</b></td></tr>
        </tbody></table></div>
        <p class="note">Weather: ${W.fitted ? esc(W.note) : 'not yet in the totals backtest. ' + esc(W.note)}</p>
      </div>

      <div class="card">
        <h2>What was tested</h2>
        <p class="lead">Each input below was added to (or removed from) the final spread model and the whole leave-one-season-out backtest was rerun (fitted on ${T.n_games.toLocaleString()} games from 2019-2025, scored on 2022-2025). Final model: average miss ${T.final.mae.toFixed(2)}, ${pct(T.final.ats, 1)} against the spread, ${pct(T.final.ats5, 1)} at 5+, ${pct(T.final.ats7, 1)} at 7+.</p>
        <div class="tblwrap"><table><thead><tr><th>Input</th><th class="num">Avg miss</th><th class="num">ATS</th><th class="num">5+ (n)</th><th class="num">7+ (n)</th><th>Verdict</th></tr></thead><tbody>
          <tr><td colspan="6" class="dim">Removing an input from the final model</td></tr>
          ${T.removed.map((x) => testRow(x, 'removed')).join('')}
          <tr><td colspan="6" class="dim">Adding an input to the final model</td></tr>
          ${T.added.map((x) => testRow(x, 'added')).join('')}
        </tbody></table></div>
        <h3 style="margin-top:16px">Does it beat the closing line by itself?</h3>
        <p class="lead" style="margin-top:0">A separate check: regress the cover margin (result vs the closing line) on each situational factor. If the market already prices something, its effect here is near zero. A t-value beyond about ±2 is a real signal.</p>
        <div class="tblwrap"><table><thead><tr><th>Factor</th><th class="num">Points vs line</th><th class="num">t-value</th></tr></thead><tbody>
          ${T.vs_line.map((x) => `<tr><td>${esc(x.label)}</td><td class="num">${sgn(x.pts, 2)}</td><td class="num ${Math.abs(x.t) >= 2 ? 'good' : 'dim'}">${sgn(x.t, 1)}</td></tr>`).join('')}
        </tbody></table></div>
        <p class="note">Takeaways: travel, rest, byes, night games, altitude and turnover margin are already in the closing number. QB news is the exception, but only when it is fresh. A QB change the market has already seen in a game is, if anything, overpriced. That is why the QB switch on the Game Detail tab is manual and meant for late news.</p>
      </div>

      <div class="card prose">
        <h2>Data and refresh</h2>
        <ul>
          <li><b>Daily at 6:45 AM Central</b> (plus Saturday 10:15 AM and Friday afternoon): new plays, updated ratings, DraftKings lines (1 Odds API call, 2 credits), and weather for outdoor games inside the 16-day forecast.</li>
          <li><b>Line history:</b> each game keeps the first DraftKings line the job sees and the last one before kickoff, so you can see whether the market moved toward the model.</li>
          <li><b>Retrain:</b> run the "Retrain model" workflow each August (or any time). It rebuilds 2019 to now, refits every weight, reruns this backtest and refits the weather values.</li>
          <li><b>Not used:</b> betting splits and injury feeds. College injury reporting is too inconsistent to automate.</li>
          <li><b>Sources:</b> SportsDataverse cfbfastR data (play-by-play, ESPN lines and FPI, schedules), CollegeFootballData.com (stadium locations), The Odds API (DraftKings), Open-Meteo (weather).</li>
        </ul>
        <p class="muted">Glossary. <b>Success rate:</b> share of plays that gain enough to stay on schedule (measured with EPA > 0). <b>EPA:</b> expected points added on a play, given down, distance and field position. <b>FPI:</b> ESPN's Football Power Index game projection. <b>Edge:</b> distance between model and line in points. <b>Cover chance:</b> backtested chance the model's side covers.</p>
      </div>`;
  }

  // ---------- boot ----------
  async function boot() {
    try {
      const [site, week, ratings, results, model, backtest, tests] = await Promise.all(
        ['site.json', 'week.json', 'ratings.json', 'results.json', 'model.json', 'backtest.json', 'tests.json'].map(load));
      Object.assign(S, { site, week, ratings, results, model, backtest, tests });
      S.ratingsById = {}; ratings.teams.forEach((t) => (S.ratingsById[t.id] = t));
    } catch (e) {
      $('#tab-week').innerHTML = `<div class="card empty">Could not load data (${esc(e.message)}). If this is a fresh deploy, run the daily workflow once.</div>`;
      return;
    }
    document.querySelectorAll('.tab').forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
    window.addEventListener('hashchange', () => {
      const x = (location.hash || '').replace('#', '');
      if (x.startsWith('detail-')) { if (String(UI.gameId) !== x.slice(7) || UI.tab !== 'detail') { UI.gameId = x.slice(7); UI.custom = null; setTab('detail'); } }
      else if (['week', 'detail', 'results', 'ratings', 'method'].includes(x) && x !== UI.tab) setTab(x);
    });
    const h = (location.hash || '').replace('#', '');
    if (h.startsWith('detail-')) { UI.gameId = h.slice(7); setTab('detail'); } else if (['week', 'detail', 'results', 'ratings', 'method'].includes(h)) setTab(h); else setTab('week');
  }
  boot();
})();
