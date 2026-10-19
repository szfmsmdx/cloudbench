(() => {
  const number = value => value == null ? '—' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  const elapsed = ms => ms < 60000 ? `${number(ms / 1000)}s` : `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}`;
  const compact = value => value >= 1e6 ? `${number(value / 1e6)}M` : value >= 1000 ? `${number(value / 1000)}K` : number(value);
  const blue = '#3984c6', green = '#2b9976', red = '#d36d65', amber = '#c8983d';
  const definitions = [
    { id: 'qps', title: '请求速率', unit: 'req/s', rate: true, series: [
      { key: 'sent_qps', name: '发起', color: blue },
      { key: 'success_qps', name: '成功完成', color: green },
      { key: 'failed_qps', name: '失败', color: red }
    ] },
    { id: 'tokens', title: '输出吞吐 · usage 上报', unit: 'tok/s', rate: true, series: [
      { key: 'reported_output_tps', name: '已报告输出', color: green }
    ] },
    { id: 'inflight', title: '在途请求', unit: '个', series: [
      { key: 'inflight', name: '等待或生成中', color: blue }
    ] },
    { id: 'latency', title: '完成请求 · 首输出延迟', unit: 'ms', series: [
      { key: 'ttft_p50_ms', name: 'p50', color: green },
      { key: 'ttft_p95_ms', name: 'p95', color: amber }
    ] }
  ];

  function mount() {
    let section = document.querySelector('#run-timeline');
    if (section) return section;
    document.querySelector('#result-metrics').insertAdjacentHTML('afterend', `
      <section id="run-timeline" class="timeline-section" aria-label="测试时间序列">
        <div class="timeline-heading"><div><strong><i class="timeline-dot"></i>运行曲线</strong><span id="timeline-status"></span></div><a id="export-timeline" class="timeline-export">时间序列 CSV ↓</a></div>
        <div id="timeline-summary" class="timeline-summary"></div>
        <div class="timeline-grid">${definitions.map(d => `<article class="timeline-card"><div class="timeline-card-title"><h4>${d.title}</h4><span>${d.unit}</span></div><div id="chart-${d.id}" class="chart-plot"></div></article>`).join('')}</div>
        <p class="timeline-caption">每秒采样；末尾不足 1 秒时合并前一窗口计算速率。输出 token 在 usage 上报时计入；片段/s 表示流式输出活跃度。</p>
      </section>`);
    section = document.querySelector('#run-timeline');
    section.addEventListener('pointermove', event => {
      const plot = event.target.closest('.chart-plot');
      if (!plot?._chart) return;
      const { points, definition, endMs } = plot._chart;
      if (!points.length) return;
      const rect = plot.querySelector('svg').getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width * 760;
      const time = Math.max(0, Math.min(1, (x - 58) / 682)) * endMs;
      let low = 0, high = points.length - 1;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (points[middle].elapsed_ms < time) low = middle + 1;
        else high = middle;
      }
      if (low > 0 && Math.abs(points[low - 1].elapsed_ms - time) < Math.abs(points[low].elapsed_ms - time)) low--;
      const point = points[low];
      const tooltip = plot.querySelector('.chart-tooltip');
      const windowMs = definition.rate ? (point.rate_window_ms ?? point.window_ms) : point.window_ms;
      tooltip.innerHTML = `<strong>+${elapsed(point.elapsed_ms)} · 窗口 ${number(windowMs / 1000)}s</strong>${definition.series.map(s => `<span><i style="background:${s.color}"></i>${s.name}<b>${number(point[s.key])} ${definition.unit}</b></span>`).join('')}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(8, Math.min(event.clientX - plot.getBoundingClientRect().left + 12, plot.clientWidth - 210))}px`;
      const cursor = plot.querySelector('.chart-cursor');
      const cursorX = 58 + point.elapsed_ms / endMs * 682;
      cursor.setAttribute('x1', cursorX);
      cursor.setAttribute('x2', cursorX);
      cursor.style.display = '';
    });
    section.addEventListener('pointerout', event => {
      const plot = event.target.closest('.chart-plot');
      if (!plot || plot.contains(event.relatedTarget)) return;
      plot.querySelector('.chart-tooltip')?.setAttribute('hidden', '');
      const cursor = plot.querySelector('.chart-cursor');
      if (cursor) cursor.style.display = 'none';
    });
    return section;
  }

  function renderChart(plot, definition, points, target) {
    const endMs = Math.max(100, points.at(-1)?.elapsed_ms || 0);
    let max = target || 0;
    for (const p of points) for (const s of definition.series) if (Number.isFinite(p[s.key])) max = Math.max(max, p[s.key]);
    const magnitude = 10 ** Math.floor(Math.log10(max || 1));
    const scaled = (max || 1) / magnitude;
    const ceiling = (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * magnitude;
    const x = p => 58 + p.elapsed_ms / endMs * 682;
    const y = value => 178 - value / ceiling * 158;
    const grid = [0, .25, .5, .75, 1].map(ratio => `<line x1="58" y1="${y(ceiling * ratio)}" x2="740" y2="${y(ceiling * ratio)}" class="chart-gridline"/><text x="48" y="${y(ceiling * ratio) + 4}" text-anchor="end">${compact(ceiling * ratio)}</text>`).join('');
    const ticks = [0, .25, .5, .75, 1].map(ratio => `<text x="${58 + ratio * 682}" y="204" text-anchor="${ratio === 0 ? 'start' : ratio === 1 ? 'end' : 'middle'}">${elapsed(endMs * ratio)}</text>`).join('');
    const paths = definition.series.map(s => {
      let open = false;
      const path = points.map(p => {
        if (!Number.isFinite(p[s.key])) { open = false; return ''; }
        const command = open ? 'L' : 'M';
        open = true;
        return `${command}${x(p).toFixed(2)},${y(p[s.key]).toFixed(2)}`;
      }).join(' ');
      const dots = points.filter((p, i) => Number.isFinite(p[s.key]) && (!Number.isFinite(points[i - 1]?.[s.key]) || !Number.isFinite(points[i + 1]?.[s.key])))
        .map(p => `<circle cx="${x(p).toFixed(2)}" cy="${y(p[s.key]).toFixed(2)}" r="4" fill="${s.color}"/>`).join('');
      return `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke"/>${dots}`;
    }).join('');
    const last = points.at(-1);
    const legend = definition.series.map(s => `<span><i style="background:${s.color}"></i>${s.name}<b>${number(last?.[s.key])}</b></span>`).join('');
    plot.innerHTML = `<svg viewBox="0 0 760 220" role="img" aria-label="${definition.title}，按测试经过时间展示">${grid}${ticks}${target != null ? `<line x1="58" y1="${y(target)}" x2="740" y2="${y(target)}" class="chart-target"/>` : ''}${paths}<line class="chart-cursor" x1="58" y1="20" x2="58" y2="178" style="display:none"/></svg><div class="chart-legend">${legend}${target != null ? `<span class="chart-target-legend">目标 ${number(target)}</span>` : ''}</div><div class="chart-tooltip" hidden></div>${!points.length ? '<div class="chart-waiting">等待采样数据</div>' : ''}`;
    plot._chart = { definition, points, endMs };
  }

  window.renderTimeline = ({ run, timeline = [] }) => {
    const section = mount();
    const signature = `${run.id}:${run.status}:${timeline.at(-1)?.seq ?? -1}`;
    if (section.dataset.signature === signature) return;
    section.dataset.signature = signature;
    section.classList.toggle('is-live', run.status === 'running');
    document.querySelector('#timeline-status').textContent = timeline.length ? (run.status === 'running' ? '实时 · 每秒采样' : '已保存 · 可回看') : (run.status === 'running' ? '正在准备数据集' : '此记录没有时间序列');
    document.querySelector('#export-timeline').href = `/api/runs/${run.id}/timeline.csv`;
    const last = timeline.at(-1);
    document.querySelector('#timeline-summary').innerHTML = last
      ? `<span>经过 <b>${elapsed(last.elapsed_ms)}</b></span><span>发起 <b>${number(last.sent)}</b></span><span>完成 <b>${number(last.completed)}</b></span><span>在途 <b>${number(last.inflight)}</b></span><span>流式片段 <b>${number(last.stream_events_per_s)}/s</b></span><span>输出 usage <b>${number(last.usage_known)}/${number(last.completed)}</b></span>`
      : '<span>开始发压后显示实时曲线</span>';
    for (const definition of definitions) {
      const target = definition.id === 'qps' && run.mode === 'async' && run.request_rate > 0 ? run.request_rate : null;
      renderChart(document.querySelector(`#chart-${definition.id}`), definition, timeline, target);
    }
  };
})();
