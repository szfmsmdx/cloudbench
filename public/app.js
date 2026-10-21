const $ = selector => document.querySelector(selector);
const state = { targets: [], prices: [], datasets: [], runs: [], currentRun: null, poll: null };
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (value, decimals = 1) => value == null ? '—' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: decimals });
const dateTime = value => value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '—';

function toast(message, error = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('error', error);
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), 3500);
}
async function api(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function view(name) {
  document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === `view-${name}`));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === name));
  $('#view-name').textContent = ({ overview: '概览', targets: '模型端点', prices: '价格目录', history: '测试记录' })[name];
  location.hash = name;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function setSelectOptions(select, items, placeholder, label) {
  const old = select.value;
  select.innerHTML = `<option value="">${escapeHtml(placeholder)}</option>` + items.map(item => `<option value="${escapeHtml(item.id)}" ${item.ready === false ? 'disabled' : ''}>${escapeHtml(label(item))}</option>`).join('');
  if (items.some(item => item.id === old && item.ready !== false)) select.value = old;
}
function renderDatasetInfo() {
  const dataset = state.datasets.find(d => d.id === $('#run-dataset').value);
  $('#local-options').hidden = !dataset;
  const distribution = (dataset?.distribution || []).filter(row => row.count > 0);
  const largest = Math.max(1, ...distribution.map(row => row.count));
  const raw = dataset?.format === 'raw-source-v1';
  const sourceUnit = dataset?.category === 'agent' ? '个会话记录' : '条原始记录';
  const distributionCount = dataset?.request_total != null
    ? `${fmt(dataset.request_total, 0)} 个${dataset.category === 'agent' ? '逐轮' : ''}请求`
    : `${fmt(dataset?.item_count, 0)} ${dataset?.category === 'agent' ? '个会话；逐轮请求数未统计' : '个样本'}`;
  const scope = dataset?.kind?.startsWith('public:')
    ? `${dataset.complete ? '完整来源' : '未完整导入'}：${dataset.scope || dataset.preparation?.scope || '旧版抽样池'}；原始 ${fmt(dataset.source_total ?? dataset.raw_count, 0)} 条。${raw ? '按本次条数读取，长度筛选在测速开始前完成。' : `可回放 ${fmt(dataset.item_count, 0)} ${dataset.unit || '样本'}。`}${dataset.skipped_count ? `另有 ${fmt(dataset.skipped_count, 0)} 条未适配，原始记录仍保留。` : ''}` : '';
  $('#dataset-info').innerHTML = dataset
    ? `<div class="dataset-summary"><div class="dataset-summary-name"><strong>${escapeHtml(dataset.name)}</strong><span>${fmt(dataset.item_count, 0)} ${raw ? sourceUnit : escapeHtml(dataset.unit || '样本')}${dataset.category === 'agent' && dataset.request_total != null ? ` · ${fmt(dataset.request_total, 0)} 个逐轮请求` : ''}</span></div>
       <div class="dataset-hover"><button type="button" class="dataset-info-trigger" aria-describedby="dataset-distribution-popup">Token 分布 <span aria-hidden="true">ⓘ</span></button>
       <div id="dataset-distribution-popup" class="dataset-popover" role="tooltip"><div class="dataset-popover-title"><strong>输入长度分布${distribution.length ? '' : ' · 未统计'}</strong><span>${distributionCount}</span></div>
       <div class="dataset-histogram">${distribution.length ? distribution.map(row => `<div class="dataset-histogram-row"><span>${escapeHtml(row.range)}</span><span class="dataset-histogram-track" aria-hidden="true"><i style="width:${row.count / largest * 100}%"></i></span><span class="dataset-histogram-value">${fmt(row.count, 0)} · ${fmt(row.percentage, 2)}%</span></div>`).join('') : '尚未对全部记录统计 token 长度，暂无全量分布图。本次选中数据的长度筛选仍在测速前完成。'}</div>
       <small>${escapeHtml(dataset.length_label || 'Kimi K3 token')} · 1K = 1024</small></div></div></div><p>${escapeHtml(scope)}</p><p>${escapeHtml(dataset.description || '')}${dataset.category === 'agent' ? ' 条数、QPS和并发按会话计算；逐轮发送原始历史，不执行工具。' : ''}</p>`
    : '选择已准备的数据集后显示长度分布。公开数据准备方式见 README。';
  renderMode();
}
function renderMode() {
  const isAsync = $('#run-form').elements.mode.value === 'async';
  const agent = state.datasets.find(d => d.id === $('#run-dataset').value)?.category === 'agent';
  $('#async-fields').hidden = !isAsync;
  $('#traffic-note').textContent = isAsync
    ? `QPS按${agent ? '会话启动' : '请求'}计算；QPS=0尽快发起。最大并发留空不限制；填写后限制${agent ? '活跃会话数，同会话逐轮等待完成' : '在途请求数'}。${agent ? '同会话始终逐轮执行。' : ''}`
    : `同步模式串行发送${agent ? '会话，同会话逐轮回放' : '请求'}，不设置目标 QPS。`;
  document.querySelectorAll('.mode-choice').forEach(el => el.classList.toggle('selected', el.querySelector('input').checked));
}
function statusHtml(status) {
  const label = { running: '运行中', completed: '已完成', cancelled: '已停止', error: '失败' }[status] || status;
  return `<span class="status ${escapeHtml(status)}">${label}</span>`;
}
function renderState() {
  $('#stat-targets').textContent = state.targets.length;
  $('#stat-runs').textContent = state.runs.length;
  $('#stat-prices').textContent = state.prices.length;
  $('#nav-run-count').textContent = state.runs.length;
  $('#target-count').textContent = state.targets.length;
  $('#price-count').textContent = state.prices.length;
  setSelectOptions($('#run-target'), state.targets, state.targets.length ? '请选择端点' : '请先添加端点', t => `${t.name} · ${t.model}`);
  setSelectOptions($('#run-price'), state.prices, '不计算 token 费用', p => `${p.provider} / ${p.model} · ${p.currency}`);
  const selectableDatasets = state.datasets.filter(d => d.kind !== 'public:open-swe');
  setSelectOptions($('#run-dataset'), selectableDatasets, '请选择数据集', d => `${d.name} · ${d.ready === false ? (d.preparation && d.preparation.status !== 'error' ? `下载中 ${d.preparation.files_done || 0} 文件` : '未准备') : `${d.item_count} ${d.format === 'raw-source-v1' ? (d.category === 'agent' ? '个会话记录' : '条原始记录') : d.unit || '样本'}${d.kind?.startsWith('public:') && !d.complete ? '（未完整）' : ''}`}`);
  if (!$('#run-dataset').value) $('#run-dataset').value = selectableDatasets.find(d => d.ready !== false)?.id || '';
  renderDatasetInfo();
  $('#target-list').innerHTML = state.targets.length ? state.targets.map(t => `<div class="target-item"><div class="target-item-top"><strong>${escapeHtml(t.name)}</strong><span class="status ${t.has_key ? '' : 'running'}">${t.has_key ? '已设密钥' : '无密钥'}</span></div><code>${escapeHtml(t.base_url)}</code><small>${escapeHtml(t.model)}</small><div class="target-actions"><button type="button" class="mini-button" data-action="select-target" data-id="${escapeHtml(t.id)}">用此端点测试</button><button type="button" class="mini-button" data-action="set-key" data-id="${escapeHtml(t.id)}">设置密钥</button><button type="button" class="mini-button danger" data-action="delete-target" data-id="${escapeHtml(t.id)}">删除</button></div></div>`).join('') : '<div class="empty">尚无端点。填写左侧信息，即可开始连接。</div>';
  $('#price-rows').innerHTML = state.prices.length ? state.prices.map(p => `<tr><td><strong>${escapeHtml(p.provider)}</strong><small>${escapeHtml(p.model)}</small></td><td>${fmt(p.input_price, 6)}</td><td>${fmt(p.cache_read_price, 6)}</td><td>${fmt(p.cache_write_price, 6)}</td><td>${fmt(p.output_price, 6)}</td><td>${escapeHtml(p.currency)}</td><td>${escapeHtml(p.checked_at)}</td><td>${p.source_url ? `<a href="${escapeHtml(p.source_url)}" target="_blank" rel="noopener noreferrer">查看 ↗</a>` : '—'}</td></tr>`).join('') : '<tr><td colspan="8"><div class="empty">暂无价格。录入后可在创建测试时选择。</div></td></tr>';
  $('#overview-recent').innerHTML = state.runs.length ? state.runs.slice(0, 4).map(r => `<a href="#history" class="recent-item" data-run="${escapeHtml(r.id)}"><div><strong>${escapeHtml(r.target_name)}</strong><small>${escapeHtml(r.model)} · ${dateTime(r.started_at)}</small></div>${statusHtml(r.status)}</a>`).join('') : '<div class="empty">第一次测试的结果会出现在这里。</div>';
  $('#history-rows').innerHTML = state.runs.length ? state.runs.map(r => `<tr><td>${dateTime(r.started_at)}</td><td><strong>${escapeHtml(r.target_name)}</strong></td><td>${escapeHtml(r.model)}</td><td>${r.request_count}</td><td>${statusHtml(r.status)}</td><td><button type="button" class="mini-button" data-run="${escapeHtml(r.id)}">查看结果 →</button></td></tr>`).join('') : '<tr><td colspan="6"><div class="empty">暂无运行记录。</div></td></tr>';
}
async function refresh() {
  const data = await api('/api/state');
  Object.assign(state, data);
  renderState();
  clearTimeout(state.preparationPoll);
  if (state.datasets.some(d => d.preparation && !['complete', 'error'].includes(d.preparation.status))) {
    state.preparationPoll = setTimeout(() => refresh().catch(e => toast(e.message, true)), 10000);
  }
}
function metric(label, value, unit, foot) {
  return `<div class="metric-card"><span>${label}</span><strong>${value}<em>${unit}</em></strong><small>${foot}</small></div>`;
}
function renderRun(data) {
  state.currentRun = data;
  const { run, samples, summary: s } = data;
  $('#result-panel').hidden = false;
  $('#result-title').textContent = `${run.target_name} · ${run.model}`;
  $('#result-subtitle').textContent = `${dateTime(run.started_at)} · ${run.dataset_name || '旧版提示词'} · ${run.mode === 'async' ? `异步 ${run.request_rate === 0 ? '突发' : `${run.request_rate} QPS`} / 并发${run.concurrency === 0 ? '不限制' : ` ${run.concurrency}`}` : '同步串行'} · ${run.request_count} 个请求`;
  if (run.unit_count && run.unit_count !== run.request_count) $('#result-subtitle').textContent += ` · ${run.unit_count} 个会话`;
  $('#run-status').textContent = ({ running: '测试运行中', completed: '测试完成', cancelled: '测试已停止', error: '测试失败' })[run.status] || run.status;
  $('#run-progress').textContent = `${s.completed} / ${run.request_count}`;
  $('#run-progress-bar').style.width = `${Math.min(100, s.completed / run.request_count * 100)}%`;
  $('#cancel-run').hidden = run.status !== 'running';
  $('#run-button').disabled = run.status === 'running';
  $('#export-summary').href = `/api/runs/${run.id}/summary.csv`;
  $('#export-samples').href = `/api/runs/${run.id}/samples.csv`;
  const currency = run.price_snapshot ? JSON.parse(run.price_snapshot).currency : '';
  $('#result-metrics').innerHTML = [
    metric('TTFT · p50', fmt(s.ttft.p50), 'ms', `${s.ttft.n} 个有效样本`),
    metric('TPOT · p50', fmt(s.tpot.p50), 'ms', `${s.tpot.n} 个有效样本`),
    metric('E2E · p95', fmt(s.e2e.p95), 'ms', `${s.e2e.n} 个有效样本`),
    metric('输出吞吐', fmt(s.output_tps), 'tok/s', '测试窗口总吞吐'),
    metric('请求吞吐', fmt(s.request_tps), 'req/s', run.mode === 'async' ? (run.request_rate === 0 ? '突发模式' : `目标 ${run.request_rate} QPS`) : '同步串行'),
    metric('实际到达速率', fmt(s.actual_arrival_qps), 'QPS', run.request_rate === 0 ? '突发模式不计算' : '按请求发起时间计算'),
    metric('缓存命中', s.cache_hit_rate == null ? '—' : fmt(s.cache_hit_rate), '%', `${s.cache_coverage} 个有缓存数据的请求`),
    metric('成功率', s.success_rate == null ? '—' : fmt(s.success_rate), '%', `${s.success} 成功 / ${s.failed} 失败`),
    metric('费用估算', s.estimated_cost == null ? '—' : fmt(s.estimated_cost, 6), currency, `${s.price_coverage} 个请求可计价`),
    metric('卡时成本', s.hardware_hourly_cost == null ? '—' : fmt(s.hardware_hourly_cost, 4), `${run.gpu_currency || ''}/h`, run.gpu_count ? `${run.gpu_count} 张 GPU` : '填写 GPU 数量和单价'),
    metric('本次硬件成本', s.hardware_run_cost == null ? '—' : fmt(s.hardware_run_cost, 6), run.gpu_currency || '', s.hardware_per_1m_output == null ? '尚无输出 token 数据' : `每百万输出 token：${fmt(s.hardware_per_1m_output, 2)}`)
  ].join('');
  let note = 'TTFT 为客户端观测的首个有效输出片段；TPOT 需服务端返回输出 token 用量。';
  if (run.dataset_id?.startsWith('public:')) {
    const snapshot = JSON.parse(run.dataset_snapshot || '{}');
    note += ` 长度口径：${snapshot.length_label || '未知'}；实际token以服务端usage为准。`;
    if (run.unit_count !== run.request_count) note += ' 固定轨迹仅测性能，生成结果不接入后续历史；到达率按会话首轮统计。';
  }
  if (s.cache_hit_rate == null) note += ' 当前端点未提供可用的缓存 token 用量，命中率显示未知。';
  if (s.estimated_cost == null && run.price_snapshot) note += ' 费用信息不完整，暂不计算总价。';
  if (s.hardware_run_cost != null) note += ' 硬件成本按整个测试窗口的全部 GPU 计费，受当前并发和输出长度影响。';
  if (run.mode === 'async' && s.queue_delay_p50_ms != null) note += ` 请求排队延迟 p50：${fmt(s.queue_delay_p50_ms)} ms。`;
  if (run.error) note += ` 运行错误：${run.error}`;
  $('#result-note').textContent = note;
  renderTimeline(data);
  $('#sample-rows').innerHTML = samples.length ? samples.map(row => `<tr><td>${row.seq}</td><td>${statusHtml(row.status)}${row.error ? `<small title="${escapeHtml(row.error)}">${escapeHtml(row.error.slice(0, 45))}</small>` : ''}</td><td>${fmt(row.ttft_ms)} ms</td><td>${fmt(row.tpot_ms)} ms</td><td>${fmt(row.e2e_ms)} ms</td><td>${fmt(row.prompt_tokens, 0)}</td><td>${fmt(row.completion_tokens, 0)}</td><td>${fmt(row.cached_tokens, 0)}</td><td>${fmt(row.estimated_cost, 6)}</td></tr>`).join('') : '<tr><td colspan="9"><div class="empty">正在等待第一个请求完成…</div></td></tr>';
}
async function loadRun(id, navigate = true) {
  clearTimeout(state.poll);
  const version = state.runLoadVersion = (state.runLoadVersion || 0) + 1;
  const data = await api(`/api/runs/${id}`);
  if (version !== state.runLoadVersion) return;
  renderRun(data);
  if (navigate) {
    view('targets');
    $('#result-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  if (data.run.status === 'running') {
    const poll = async () => {
      try {
        const current = await api(`/api/runs/${id}`);
        if (version !== state.runLoadVersion) return;
        renderRun(current);
        if (current.run.status === 'running') {
          state.poll = setTimeout(poll, 650);
        } else {
          refresh().catch(e => toast(e.message, true));
        }
      } catch (e) { if (version === state.runLoadVersion) toast(e.message, true); }
    };
    state.poll = setTimeout(poll, 650);
  }
}

document.addEventListener('click', async event => {
  const v = event.target.closest('[data-view]');
  if (v) { view(v.dataset.view); return; }
  const runLink = event.target.closest('[data-run]');
  if (runLink) { event.preventDefault(); try { await loadRun(runLink.dataset.run); } catch (e) { toast(e.message, true); } return; }
  const action = event.target.closest('[data-action]');
  if (!action) return;
  const id = action.dataset.id;
  if (action.dataset.action === 'select-target') { $('#run-target').value = id; $('#run-panel').scrollIntoView({ behavior: 'smooth' }); return; }
  if (action.dataset.action === 'set-key') {
    const key = prompt('输入此端点的 API Key（只保存在当前服务进程内存中）');
    if (!key) return;
    try { await api(`/api/targets/${id}/key`, { method: 'POST', body: JSON.stringify({ api_key: key }) }); await refresh(); toast('API Key 已设置'); } catch (e) { toast(e.message, true); }
  }
  if (action.dataset.action === 'delete-target') {
    if (!confirm('删除这个端点？已有测试记录仍会保留。')) return;
    try { await api(`/api/targets/${id}`, { method: 'DELETE' }); await refresh(); toast('端点已删除'); } catch (e) { toast(e.message, true); }
  }
});
$('#hero-start').addEventListener('click', () => { view('targets'); $('#run-panel').scrollIntoView({ behavior: 'smooth' }); });
$('#target-form').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  try { const target = await api('/api/targets', { method: 'POST', body: JSON.stringify(data) }); form.reset(); await refresh(); $('#run-target').value = target.id; toast('端点已保存'); } catch (e) { toast(e.message, true); }
});
$('#price-form').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  try { const price = await api('/api/prices', { method: 'POST', body: JSON.stringify(data) }); form.reset(); $('#price-date').value = new Date().toISOString().slice(0, 10); await refresh(); $('#run-price').value = price.id; toast('价格版本已保存'); } catch (e) { toast(e.message, true); }
});
$('#run-dataset').addEventListener('change', renderDatasetInfo);
function positionDatasetPopover(event) {
  const trigger = event.target.closest('.dataset-info-trigger');
  if (!trigger) return;
  const rect = trigger.getBoundingClientRect();
  const popup = $('#dataset-distribution-popup');
  trigger.parentElement.style.setProperty('--popover-right', '0px');
  trigger.parentElement.classList.toggle('opens-up', window.innerHeight - rect.bottom < popup.offsetHeight + 16 && rect.top > popup.offsetHeight + 16);
  const bounds = trigger.closest('.panel').getBoundingClientRect();
  const popupRect = popup.getBoundingClientRect();
  const left = Math.max(12, bounds.left + 8);
  const right = Math.min(window.innerWidth - 12, bounds.right - 8);
  const adjustment = popupRect.left < left ? popupRect.left - left : popupRect.right > right ? popupRect.right - right : 0;
  trigger.parentElement.style.setProperty('--popover-right', `${adjustment}px`);
}
$('#dataset-info').addEventListener('pointerover', positionDatasetPopover);
$('#dataset-info').addEventListener('focusin', positionDatasetPopover);
document.querySelectorAll('input[name="mode"]').forEach(el => el.addEventListener('change', renderMode));
$('#run-form').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const data = Object.fromEntries(new FormData(form));
  if (!Number.isSafeInteger(Number(data.request_count)) || Number(data.request_count) < 1) {
    toast('请填写本次测试条数或 Agent 会话数，不能留空运行全量', true);
    return;
  }
  if (data.dataset_id === 'public:open-swe') {
    toast('Open-SWE 已从网页测试选项中移除，请选择其他数据集', true);
    return;
  }
  data.include_usage = form.elements.include_usage.checked;
  data.max_input_tokens = Number(data.context_limit_k) * 1024;
  delete data.context_limit_k;
  const button = form.querySelector('button[type="submit"]');
  const label = button.innerHTML;
  button.disabled = true;
  button.textContent = '准备本次数据…';
  try {
    const run = await api('/api/runs', { method: 'POST', body: JSON.stringify(data) });
    await refresh();
    await loadRun(run.id, true);
    toast('测试已开始');
  } catch (e) { toast(e.message, true); }
  finally { button.disabled = false; button.innerHTML = label; }
});
$('#cancel-run').addEventListener('click', async () => {
  const id = state.currentRun?.run.id;
  if (!id) return;
  try { await api(`/api/runs/${id}/cancel`, { method: 'POST' }); toast('正在停止测试'); } catch (e) { toast(e.message, true); }
});
$('#price-date').value = new Date().toISOString().slice(0, 10);
renderMode();
refresh().then(() => {
  const requested = location.hash.slice(1);
  if (['overview', 'targets', 'prices', 'history'].includes(requested)) view(requested);
  const running = state.runs.find(r => r.status === 'running');
  if (running) loadRun(running.id, false).catch(e => toast(e.message, true));
}).catch(e => toast(e.message, true));
