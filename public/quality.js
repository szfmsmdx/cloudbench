let qualityCurrent = null;
let qualityPoll = null;

function qualityThinkingLabel(run) {
  const kwargs = run.chat_template_kwargs || {};
  if (kwargs.enable_thinking === false || kwargs.thinking === false || kwargs.reasoning_effort === 'none') return '思考关闭';
  const effort = kwargs.reasoning_effort;
  const label = ({50: 'low', 62: 'medium', 75: 'high', 100: 'max'})[effort] || effort || '默认';
  return `思考 · ${label}`;
}

function renderQualityHistory(runs) {
  $('#quality-history').innerHTML = runs.length ? runs.map(run => {
    const s = run.summary;
    const complete = run.status === 'completed' && s.completed === run.count;
    const progress = Math.min(100, s.completed / Math.max(run.count, 1) * 100);
    const selected = run.id === qualityCurrent;
    return `<button type="button" class="quality-history-card${selected ? ' is-selected' : ''}" data-quality-run="${escapeHtml(run.id)}" aria-pressed="${selected}">
      <span class="quality-history-top"><strong>${escapeHtml(run.target_name)}</strong>${statusHtml(run.status)}</span>
      <span class="quality-history-model">${escapeHtml(run.model)}</span>
      <span class="quality-history-tags"><span>GPQA-Diamond</span><span>${escapeHtml(qualityThinkingLabel(run))}</span><span>并发 ${fmt(run.concurrency ?? 1, 0)}</span></span>
      <span class="quality-history-score"><span><strong>${fmt(s.accuracy, 2)}<em>%</em></strong><small>${complete ? '正确率' : '阶段正确率'}</small></span><span class="quality-history-count"><b>${fmt(s.correct, 0)}</b> 题答对<small>${fmt(s.completed, 0)} / ${fmt(run.count, 0)} 题已完成</small></span></span>
      <span class="quality-history-progress" aria-hidden="true"><i style="width:${progress}%"></i></span>
      <span class="quality-history-bottom"><time datetime="${escapeHtml(run.started_at)}">${escapeHtml(dateTime(run.started_at))}</time><span>查看结果 <span aria-hidden="true">↗</span></span></span>
    </button>`;
  }).join('') : '<div class="empty">暂无跑分记录。每次评测运行全量 198 题。</div>';
}

function qualityEstimate() {
  const form = $('#quality-form');
  const count = 198;
  const budget = count * Number(form.elements.max_tokens.value);
  const price = state.prices.find(p => p.id === form.elements.price_id.value);
  $('#quality-estimate').textContent = `本次 ${count} 题，最多 ${fmt(budget, 0)} 输出 tokens。` + (price
    ? `按所选报价，输出部分预算约 ${fmt(budget * price.output_price / 1e6, 4)} ${price.currency}，另加输入费用；实际按 usage 计算。`
    : '选择价格版本可估算费用；未选择时显示 token 用量，不把未知费用当成零。');
}

window.refreshQuality = async function () {
  setSelectOptions($('#quality-target'), state.targets, '请选择端点', t => `${t.name} · ${t.model}${t.has_key ? '' : ' · 未设置密钥'}`);
  setSelectOptions($('#quality-price'), state.prices, '不估算费用（仍记录 token 用量）', p => `${p.provider} / ${p.model} · ${p.currency}`);
  const data = await api('/api/quality');
  $('#quality-ready').textContent = data.ready ? 'GPQA-Diamond 已准备；只把问题和选项发送给模型，标准答案留在本地判分。' : '数据未准备，请管理员运行 python -m pybench.gpqa。';
  renderQualityHistory(data.runs);
  qualityEstimate();
  if (!qualityCurrent && data.runs.length) await loadQuality(data.runs[0].id);
};

async function loadQuality(id) {
  clearTimeout(qualityPoll);
  const run = await api(`/api/quality/${id}`);
  qualityCurrent = id;
  document.querySelectorAll('[data-quality-run]').forEach(card => {
    const selected = card.dataset.qualityRun === id;
    card.classList.toggle('is-selected', selected);
    card.setAttribute('aria-pressed', String(selected));
  });
  const s = run.summary;
  $('#quality-result').hidden = false;
  $('#quality-title').textContent = `${run.target_name} · GPQA-Diamond`;
  const thinking = qualityThinkingLabel(run);
  $('#quality-progress').textContent = `${({running:'运行中',completed:'已完成',cancelled:'已停止',error:'中断'})[run.status]} · ${s.completed} / ${s.total} 题 · seed ${run.seed} · ${thinking} · 并发 ${run.concurrency ?? 1} · 每题上限 ${run.max_tokens} tokens`;
  $('#quality-cancel').hidden = run.status !== 'running';
  $('#quality-export').href = `/api/quality/${id}/export`;
  $('#quality-metrics').innerHTML = [metric('正确率', fmt(s.accuracy, 2), '%', `${s.correct} / ${s.completed} 已完成题目，含失败与截断`),
    metric('95% 置信区间', s.confidence_interval ? s.confidence_interval.map(x => fmt(x, 2)).join('–') : '—', '%', 'Wilson 区间；抽样或未完成结果仅供参考'),
    metric('输出用量', fmt(s.completion_tokens, 0), 'tokens', `${s.usage_coverage} / ${s.completed} 题有输出 usage`),
    metric('费用估算', fmt(s.estimated_cost, 5), escapeHtml(s.currency || ''), s.estimated_cost == null ? '未选择报价或 usage 不完整' : '按本次保存的价格快照计算')].join('');
  $('#quality-note').textContent = `请求失败 ${s.errors} · 无法解析 ${s.unparsed} · 输出截断 ${s.truncated}。上述均计为未答对；未开始或停止中的题不进入分母。识别最终 ANSWER: A/B/C/D 行（兼容 Markdown 加粗）。${run.count < 198 ? '本次为抽样。' : ''}自定义生成式评测，不直接对比其他提示词或框架的榜单。${run.error || ''}`;
  $('#quality-samples').innerHTML = run.samples.map(row => `<tr><td>${row.seq}<small>源题 ${row.source_index + 1}</small></td><td>${row.correct ? '✓ 正确' : ({scored:'✗ 错误',error:'请求失败',unparsed:'无法解析',truncated:'输出截断'})[row.status]}</td><td>${escapeHtml(row.answer || '—')}</td><td>${escapeHtml(row.expected)}</td><td>${fmt(row.completion_tokens, 0)}</td><td><details><summary>查看题目与回答</summary><pre style="white-space:pre-wrap;max-width:600px;max-height:400px;overflow:auto">${escapeHtml(row.question)}\n\n${escapeHtml(Object.entries(row.options).map(([k,v]) => `${k}. ${v}`).join('\n'))}\n\n模型回答：\n${escapeHtml(row.response)}${row.reasoning ? '\n\n推理：\n' + escapeHtml(row.reasoning) : ''}${row.error ? '\n错误：' + escapeHtml(row.error) : ''}</pre></details></td></tr>`).join('');
  if (run.status === 'running') qualityPoll = setTimeout(async () => {
    try {
      await loadQuality(id);
      if ($('#quality-cancel').hidden) await refreshQuality();
    } catch (e) { toast(e.message, true); }
  }, 2000);
}

$('#quality-form').addEventListener('input', qualityEstimate);
$('#quality-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  try {
    const data = Object.fromEntries(new FormData(event.currentTarget));
    const thinkingMode = data.thinking_mode || 'thinking';
    data.chat_template_kwargs = thinkingMode === 'chat'
      ? {enable_thinking: false}
      : {thinking_mode: 'thinking', reasoning_effort: Number(data.reasoning_effort || 75)};
    data.concurrency = Number(data.concurrency || 16);
    delete data.thinking_mode;
    delete data.reasoning_effort;
    const target = state.targets.find(t => t.id === data.target_id);
    if (target && !target.has_key) toast('若端点需要认证，请先在模型端点中设置密钥');
    const run = await api('/api/quality', {method:'POST', body:JSON.stringify(data)});
    qualityCurrent = run.id;
    await loadQuality(run.id);
    await refreshQuality();
  } catch (e) { toast(e.message, true); }
  finally { button.disabled = false; }
});
$('#quality-cancel').addEventListener('click', async () => {
  try {
    await api(`/api/quality/${qualityCurrent}/cancel`, {method:'POST'});
    await loadQuality(qualityCurrent);
    await refreshQuality();
  } catch (e) { toast(e.message, true); }
});
$('#quality-history').addEventListener('click', event => {
  const button = event.target.closest('[data-quality-run]');
  if (button) loadQuality(button.dataset.qualityRun).catch(e => toast(e.message, true));
});
