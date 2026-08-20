/* clipbot UI — no build step, no framework. */
'use strict';

const $  = (id) => document.getElementById(id);
const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

const state = {
  video: null,     // {id, name, size, duration, width, height, has_audio}
  audio: null,     // {id, name}
  jobId: null,
  poll: null,
  aspect: '9:16',
  uploading: false,
};

/* ── helpers ─────────────────────────────────────────────────────── */

function hms(s) {
  if (s == null || !isFinite(s)) return '—';
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(x).padStart(2, '0')}`
           : `${m}:${String(x).padStart(2, '0')}`;
}

function mb(bytes) {
  if (!bytes) return '—';
  const gb = bytes / 1073741824;
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(bytes / 1048576).toFixed(1)} MB`;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* empty or non-JSON */ }
  if (!r.ok) {
    throw new Error((body && (body.detail || body.message)) || `${r.status} ${r.statusText}`);
  }
  return body;
}

/* ── health ──────────────────────────────────────────────────────── */

async function checkHealth() {
  try {
    const h = await api('/api/health');
    state.health = h;
    $('health').innerHTML = h.ready
      ? `<span class="pill"><span class="dot"></span>ready · ${h.free_gb} GB free</span>`
      : `<span class="pill"><span class="dot bad"></span>not ready</span>
         <span class="problem">${h.missing.join(' and ')} missing —
         run <code>brew install ffmpeg</code> and restart</span>`;
    $('foot-data').textContent = `data: ${h.data_dir}`;
    if (!h.ready) { $('go').disabled = true; $('go-note').textContent = 'ffmpeg is not installed'; }
  } catch {
    $('health').innerHTML = `<span class="pill"><span class="dot bad"></span>server unreachable</span>`;
  }
}

/* ── upload ──────────────────────────────────────────────────────── */

function upload(file, kind, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    // A raw PUT keeps the whole file out of a multipart buffer, and XHR is
    // still the only way to get real upload progress in a browser.
    xhr.open('PUT', `/api/upload?kind=${kind}&name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total, e.loaded, e.total);
    };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error((body && body.detail) || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('upload failed — is the server still running?'));
    xhr.onabort  = () => reject(new Error('upload cancelled'));
    xhr.send(file);
  });
}

async function takeVideo(file) {
  if (state.uploading) return;
  state.uploading = true;

  $('drop').classList.add('hidden');
  $('source').classList.add('hidden');
  $('upload').classList.remove('hidden');
  $('upload-name').textContent = file.name;
  $('upload-bar').style.width = '0%';
  $('upload-pct').textContent = '0%';
  $('upload-note').textContent = `uploading ${mb(file.size)}`;

  const t0 = performance.now();
  try {
    const info = await upload(file, 'video', (frac, loaded) => {
      const pct = Math.round(frac * 100);
      $('upload-bar').style.width = pct + '%';
      $('upload-pct').textContent = pct + '%';
      const secs = (performance.now() - t0) / 1000;
      const rate = loaded / Math.max(secs, .3);
      $('upload-note').textContent =
        `${mb(loaded)} of ${mb(file.size)} · ${mb(rate)}/s`;
    });
    $('upload-pct').textContent = '100%';
    $('upload-note').textContent = 'reading the file…';
    state.video = info;
    showSource(info);
  } catch (e) {
    $('upload').classList.add('hidden');
    $('drop').classList.remove('hidden');
    alert(e.message);
  } finally {
    // Clear the flag before the last refresh: showSource() also calls
    // refreshGo(), and if it runs while this is still set the button stays
    // disabled with nothing left to re-enable it.
    state.uploading = false;
    refreshGo();
  }
}

function showSource(info) {
  $('upload').classList.add('hidden');
  $('drop').classList.add('hidden');
  $('source').classList.remove('hidden');
  $('source-name').textContent = info.name;

  const bits = [hms(info.duration), `${info.width}×${info.height}`,
                `${info.fps} fps`, mb(info.size)];
  if (!info.has_audio) bits.push('no audio track');
  $('source-meta').textContent = bits.join('  ·  ');

  $('step-config').classList.remove('locked');
  $('step-config').removeAttribute('aria-disabled');

  // A source with no audio cannot be a highlight clip worth watching, so point
  // at the mode that supplies its own.
  if (!info.has_audio) {
    document.querySelector('input[name=mode][value=music]').checked = true;
    syncMode();
  }
  suggest();
  refreshGo();
}

function clearSource() {
  state.video = null;
  $('source').classList.add('hidden');
  $('drop').classList.remove('hidden');
  $('file').value = '';
  $('step-config').classList.add('locked');
  $('step-config').setAttribute('aria-disabled', 'true');
  refreshGo();
}

/* ── settings ────────────────────────────────────────────────────── */

function syncMode() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  document.querySelectorAll('.mode').forEach((el) =>
    el.classList.toggle('selected', el.querySelector('input').checked));
  $('music-slot').classList.toggle('hidden', mode !== 'music');
  refreshGo();
}

function usable() {
  const v = state.video;
  if (!v || !v.duration) return null;
  return Math.max(v.duration - (+$('skip-intro').value || 0)
                             - (+$('skip-outro').value || 0), 0);
}

function suggest() {
  const v = state.video;
  if (!v || !v.duration) { $('count-hint').textContent = 'spread across the runtime'; return; }
  const count = +$('count').value, length = +$('length').value;
  const usable_ = usable();
  if (usable_ < length) {
    $('count-hint').textContent = `only ${hms(usable_)} of footage — too short for ${length}s`;
    return;
  }
  const every = usable_ / Math.max(count, 1);
  $('count-hint').textContent =
    `about one every ${hms(every)} of ${hms(usable_)}  ·  ${hms(count * length)} total`;
}

function refreshGo() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  const ready = state.health ? state.health.ready : true;
  let note = '', ok = true;

  if (!ready)                       { ok = false; note = 'ffmpeg is not installed'; }
  else if (!state.video)            { ok = false; note = 'add a video first'; }
  else if (mode === 'music' && !state.audio) { ok = false; note = 'beat-synced mode needs a track'; }
  else {
    const n = +$('count').value, len = +$('length').value;
    const room = usable();
    if (!(n >= 1 && n <= 40)) { ok = false; note = 'clip count must be 1–40'; }
    else if (!(len >= 3 && len <= 180)) { ok = false; note = 'length must be 3–180 seconds'; }
    else if (room !== null && room < len) {
      // Better to say so here than to queue a job the pipeline will reject.
      ok = false;
      note = `only ${hms(room)} of footage after the trims — shorten the clips`;
    } else {
      note = `${n} clip${n === 1 ? '' : 's'} · ${len}s each · ${$('aspect').value}`;
    }
  }
  $('go').disabled = !ok || state.uploading;
  $('go-note').textContent = note;
}

/* ── run ─────────────────────────────────────────────────────────── */

async function start() {
  const payload = {
    video_id: state.video.id,
    audio_id: state.audio ? state.audio.id : null,
    mode: document.querySelector('input[name=mode]:checked').value,
    count: +$('count').value,
    length: +$('length').value,
    aspect: $('aspect').value,
    frame: $('frame').value,
    zoom: +$('zoom').value || 1,
    quality: $('quality').value,
    auto_skip: $('auto-skip').checked,
    skip_intro: +$('skip-intro').value || 0,
    skip_outro: +$('skip-outro').value || 0,
    sharpen: $('sharpen').checked,
    normalize_audio: $('normalize').checked,
    seed: $('seed').value ? +$('seed').value : null,
  };
  state.aspect = payload.aspect;

  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.jobId = job.id;
    $('step-config').classList.add('locked');
    $('step-out').classList.add('hidden');
    $('step-run').classList.remove('hidden');
    $('run-err').classList.add('hidden');
    $('run-bar').style.width = '0%';
    $('cancel').classList.remove('hidden');
    $('step-run').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    render(job);
    state.poll = setInterval(tick, 700);
  } catch (e) {
    alert(e.message);
  }
}

async function tick() {
  if (!state.jobId) return;
  try {
    render(await api(`/api/jobs/${state.jobId}`));
  } catch {
    /* a dropped poll is not fatal — the next one will catch up */
  }
}

const STAGE_ORDER = ['scan', 'select', 'render', 'done'];
const STAGE_ALIAS = { probe: 'scan', audio: 'scan', shots: 'scan',
                      rank: 'select', starting: 'scan' };

function render(job) {
  const pct = Math.round(job.progress * 100);
  $('run-bar').style.width = pct + '%';
  $('run-msg').textContent = job.message || job.stage;
  $('run-title').textContent =
    job.state === 'running' ? `Working… ${pct}%`
    : job.state === 'done'  ? 'Done'
    : job.state === 'error' ? 'Failed'
    : job.state === 'cancelled' ? 'Cancelled' : 'Queued';

  const parts = [`${hms(job.elapsed)} elapsed`];
  if (job.eta != null && job.state === 'running') parts.push(`~${hms(job.eta)} left`);
  $('run-time').textContent = parts.join(' · ');

  const now = STAGE_ALIAS[job.stage] || job.stage;
  const at = STAGE_ORDER.indexOf(now);
  document.querySelectorAll('#stages li').forEach((li) => {
    const i = STAGE_ORDER.indexOf(li.dataset.stage);
    li.classList.toggle('at', i === at);
    li.classList.toggle('past', at > -1 && i < at);
  });

  if (job.state === 'error') {
    $('run-err').textContent = job.error;
    $('run-err').classList.remove('hidden');
  }

  if (['done', 'error', 'cancelled'].includes(job.state)) {
    clearInterval(state.poll);
    state.poll = null;
    $('cancel').classList.add('hidden');
    $('step-config').classList.remove('locked');
    if (job.state === 'done' && job.clips.length) results(job);
  }
}

function results(job) {
  $('step-out').classList.remove('hidden');
  const total = job.clips.reduce((a, c) => a + c.size, 0);
  $('out-title').textContent =
    `${job.clips.length} clip${job.clips.length === 1 ? '' : 's'} · ${mb(total)}`;

  const grid = $('clips');
  grid.innerHTML = '';
  for (const c of job.clips) {
    const src = `/api/jobs/${job.id}/clips/${c.index}`;
    const el = document.createElement('article');
    el.className = 'clip';
    el.innerHTML = `
      <div class="clip-media" data-aspect="${job.aspect}">
        <span class="clip-no">${c.index}</span>
        <img alt="" src="/api/jobs/${job.id}/thumbs/${c.index}" loading="lazy">
        <button class="play" type="button" aria-label="Play clip ${c.index}">
          <span>▶</span>
        </button>
      </div>
      <div class="clip-body">
        <span class="clip-when">${hms(c.start)} → ${hms(c.end)}</span>
        <span class="clip-sub">${c.duration.toFixed(1)}s · ${mb(c.size)}</span>
        <div class="clip-tags">${c.tags.map((t) => `<span>${t}</span>`).join('')}</div>
        <a class="dl" href="${src}?download=1" download>Download</a>
      </div>`;

    // Swap the poster for a real <video> only on click; mounting ten players
    // at once makes the browser fetch ten clips nobody asked for.
    el.querySelector('.play').addEventListener('click', (ev) => {
      const media = ev.currentTarget.parentElement;
      media.innerHTML =
        `<video src="${src}" controls autoplay playsinline preload="metadata"></video>`;
    });
    grid.appendChild(el);
  }
  $('step-out').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── wiring ──────────────────────────────────────────────────────── */

function dropzone(zone, input, handler) {
  on(zone, 'click', () => input.click());
  on(zone, 'keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  on(input, 'change', () => { if (input.files[0]) handler(input.files[0]); });
  ['dragenter', 'dragover'].forEach((ev) =>
    on(zone, ev, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    on(zone, ev, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  on(zone, 'drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) handler(f);
  });
}

dropzone($('drop'), $('file'), takeVideo);

dropzone($('music-drop'), $('music-file'), async (file) => {
  $('music-barwrap').classList.remove('hidden');
  $('music-title').textContent = file.name;
  try {
    const info = await upload(file, 'audio', (frac) => {
      $('music-bar').style.width = Math.round(frac * 100) + '%';
    });
    state.audio = info;
    $('music-sub').textContent = `ready · ${mb(info.size)}`;
    $('music-barwrap').classList.add('hidden');
  } catch (e) {
    $('music-title').textContent = 'Drop the track';
    $('music-sub').textContent = e.message;
    $('music-barwrap').classList.add('hidden');
  }
  refreshGo();
});

on($('source-clear'), 'click', clearSource);
document.querySelectorAll('input[name=mode]').forEach((r) => on(r, 'change', syncMode));

document.querySelectorAll('[data-nudge]').forEach((b) => on(b, 'click', () => {
  const [id, by] = b.dataset.nudge.split(':');
  const input = $(id);
  const next = Math.min(Math.max(+input.value + +by, +input.min), +input.max);
  input.value = next;
  if (id === 'length') markChips(next);
  suggest(); refreshGo();
}));

function markChips(v) {
  document.querySelectorAll('#length-chips button').forEach((c) =>
    c.classList.toggle('on', +c.dataset.length === +v));
}
document.querySelectorAll('#length-chips button').forEach((c) => on(c, 'click', () => {
  $('length').value = c.dataset.length;
  markChips(c.dataset.length);
  suggest(); refreshGo();
}));

['count', 'length', 'skip-intro', 'skip-outro'].forEach((id) =>
  on($(id), 'input', () => { if (id === 'length') markChips($(id).value); suggest(); refreshGo(); }));
['aspect', 'frame', 'quality', 'zoom'].forEach((id) => on($(id), 'change', refreshGo));
on($('skip-intro'), 'change', () => { suggest(); refreshGo(); });
on($('skip-outro'), 'change', () => { suggest(); refreshGo(); });

on($('go'), 'click', start);

on($('cancel'), 'click', async () => {
  if (!state.jobId) return;
  $('cancel').disabled = true;
  try { await api(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' }); }
  catch { /* it may have finished in the meantime */ }
  $('cancel').disabled = false;
});

on($('zip'), 'click', () => {
  if (state.jobId) window.location = `/api/jobs/${state.jobId}/zip`;
});

on($('again'), 'click', () => {
  $('step-out').classList.add('hidden');
  $('step-run').classList.add('hidden');
  $('step-config').classList.remove('locked');
  $('step-config').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

window.addEventListener('beforeunload', (e) => {
  if (state.uploading) { e.preventDefault(); e.returnValue = ''; }
});

checkHealth();
syncMode();
refreshGo();
