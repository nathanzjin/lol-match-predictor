"use strict";

const $ = (sel) => document.querySelector(sel);
const pct = (x) => (x * 100).toFixed(1) + "%";
const signed = (x) => (x >= 0 ? "+" : "") + x.toFixed(1);

// ---------------- tabs ----------------
let seasonLoaded = false;
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "season" && !seasonLoaded) loadSeason();
  });
});

// ---------------- init: teams + meta ----------------
async function init() {
  try {
    const [teams, meta] = await Promise.all([
      fetch("/api/teams").then((r) => r.json()),
      fetch("/api/meta").then((r) => r.json()),
    ]);
    populateTeams(teams);
    $("#meta-line").textContent =
      `${meta.n_supported_teams} supported Tier-1 teams across ${meta.regions.join(", ")}`;
  } catch (e) {
    $("#predict-error").textContent = "Could not reach the API. Is the server running?";
    $("#predict-error").classList.remove("hidden");
  }
}

function populateTeams(byRegion) {
  const mk = (sel) => {
    sel.innerHTML = "";
    for (const [region, teams] of Object.entries(byRegion)) {
      if (!teams.length) continue;
      const og = document.createElement("optgroup");
      og.label = region;
      for (const t of teams) {
        const o = document.createElement("option");
        o.value = t; o.textContent = t;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
  };
  const blue = $("#blue-team"), red = $("#red-team");
  mk(blue); mk(red);
  // sensible defaults: two different marquee teams if present
  const flat = Object.values(byRegion).flat();
  if (flat.includes("T1")) blue.value = "T1";
  if (flat.includes("Gen.G")) red.value = "Gen.G";
  else if (red.options.length > 1) red.selectedIndex = 1;
}

// ---------------- predict ----------------
$("#predict-btn").addEventListener("click", runPredict);
$("#swap-btn").addEventListener("click", () => {
  const b = $("#blue-team").value;
  $("#blue-team").value = $("#red-team").value;
  $("#red-team").value = b;
  runPredict();
});

async function runPredict() {
  const blue = $("#blue-team").value, red = $("#red-team").value;
  const window = parseInt($("#window").value, 10) || 10;
  const errEl = $("#predict-error"), resEl = $("#predict-result");
  errEl.classList.add("hidden"); resEl.classList.add("hidden");
  if (blue === red) {
    errEl.textContent = "Pick two different teams.";
    errEl.classList.remove("hidden");
    return;
  }
  $("#predict-btn").textContent = "Predicting…"; $("#predict-btn").disabled = true;
  try {
    const r = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blue, red, window }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Prediction failed.");
    renderPrediction(data);
  } catch (e) {
    errEl.textContent = e.message;
    errEl.classList.remove("hidden");
  } finally {
    $("#predict-btn").textContent = "Predict"; $("#predict-btn").disabled = false;
  }
}

function rosterTable(side) {
  const rows = side.roster.map((p) =>
    `<tr><td>${p.role}</td><td>${p.player}</td>
     <td class="num">${p.elo.toFixed(0)}</td><td class="num">${p.games}</td></tr>`).join("");
  return `<table>
    <tr><th>role</th><th>player</th><th class="num">elo</th><th class="num">gms</th></tr>
    ${rows}</table>`;
}

function renderPrediction(d) {
  const b = d.blue, r = d.red;
  const bp = Math.round(d.p_blue * 100), rp = 100 - bp;
  const diffRow = (label, v) =>
    `<tr><td>${label}</td><td class="num ${v >= 0 ? "pos" : "neg"}">${signed(v)}</td></tr>`;
  const roleRows = Object.entries(d.diffs.role_dpm)
    .map(([role, v]) => diffRow(`${role} dpm form`, v)).join("");

  $("#predict-result").innerHTML = `
    <div class="card">
      <div class="prob-bar">
        <div class="b" style="width:${bp}%">${b.team} ${bp}%</div>
        <div class="r" style="width:${rp}%">${rp}% ${r.team}</div>
      </div>
      <div class="favored"><b>${d.winner}</b> favored &middot; ${pct(d.confidence)} &middot;
        <span class="muted">blue side ${b.team} carries a ~3% positional edge</span></div>

      <div class="rosters">
        <div class="roster-card blue">
          <h4>${b.team} <span class="reg">${b.region} &middot; mean Elo ${b.mean_elo.toFixed(0)} (anchored ${b.anchored_elo.toFixed(0)})</span></h4>
          ${rosterTable(b)}
        </div>
        <div class="roster-card red">
          <h4>${r.team} <span class="reg">${r.region} &middot; mean Elo ${r.mean_elo.toFixed(0)} (anchored ${r.anchored_elo.toFixed(0)})</span></h4>
          ${rosterTable(r)}
        </div>
      </div>

      <div class="diffs">
        <h3>Key signals (blue &minus; red)</h3>
        <table>
          ${diffRow("player-Elo (mean)", d.diffs.pelo_diff)}
          ${diffRow("player-Elo (region-anchored)", d.diffs.pelo_ra_diff)}
          ${diffRow("region-anchored team rating", d.diffs.relo_diff)}
          ${roleRows}
        </table>
      </div>
    </div>`;
  $("#predict-result").classList.remove("hidden");
}

// ---------------- season tracker ----------------
let seasonChart = null;

async function loadSeason() {
  const statusEl = $("#season-status");
  try {
    const r = await fetch("/api/performance").then((x) => x.json());
    if (r.status === "ready") {
      seasonLoaded = true;
      statusEl.classList.add("hidden");
      $("#season-content").classList.remove("hidden");
      renderSeason(r);
    } else if (r.status === "computing" || r.status === "idle") {
      statusEl.classList.remove("hidden");
      statusEl.innerHTML = "Computing the season track record (replaying the full game history once)&hellip; this takes ~30&ndash;60s on first load.";
      setTimeout(loadSeason, 4000);
    } else {
      statusEl.textContent = "Season track record unavailable.";
    }
  } catch (e) {
    statusEl.textContent = "Could not load season performance.";
  }
}

function metricCard(value, key, delta) {
  return `<div class="metric"><div class="v">${value}</div><div class="k">${key}</div>
    ${delta ? `<div class="delta muted">${delta}</div>` : ""}</div>`;
}

function renderSeason(d) {
  const s = d.summary.all;
  const lift = ((s.accuracy - s.baseline_acc) * 100).toFixed(1);
  $("#season-cards").innerHTML =
    metricCard(pct(s.accuracy), `accuracy on ${s.n} ${d.season_year} Tier-1 games`,
               `+${lift} pts vs blue-side baseline (${pct(s.baseline_acc)})`) +
    metricCard(s.log_loss.toFixed(3), "log loss", "lower is better") +
    metricCard(s.brier.toFixed(3), "Brier score", "lower is better") +
    metricCard(s.auc ? s.auc.toFixed(3) : "—", "ROC-AUC", "discrimination");

  $("#season-caption").textContent =
    `Walk-forward: trained on games through ${d.trained_through}, then graded on every ${d.season_year} ` +
    `Tier-1 matchup it had never seen (data through ${d.data_through}). Each point is the cumulative score after that game.`;

  drawChart(d.timeline);

  // by-league bars
  const maxN = Math.max(...d.by_league.map((b) => b.n));
  $("#by-league").innerHTML = d.by_league.filter((b) => b.n >= 10).map((b) =>
    `<div class="bar-row">
       <span>${b.league} <span class="muted">(${b.n})</span></span>
       <div class="bar-track"><div class="bar-fill" style="width:${(b.n / maxN) * 100}%"></div></div>
       <span class="num">${pct(b.accuracy)}</span>
     </div>`).join("");

  // recent games
  $("#recent-games").innerHTML = `<table>
    <tr><th>date</th><th>matchup</th><th class="num">P(blue)</th><th>pick</th><th></th></tr>
    ${d.games.map((g) => `<tr>
      <td class="muted">${g.date}</td>
      <td>${g.blue} <span class="muted">vs</span> ${g.red}</td>
      <td class="num">${pct(g.p_blue)}</td>
      <td>${g.predicted}</td>
      <td><span class="tag ${g.correct ? "ok" : "no"}">${g.correct ? "✓" : "✗"}</span></td>
    </tr>`).join("")}
  </table>`;
}

function drawChart(timeline) {
  const ctx = $("#season-chart");
  const labels = timeline.map((t) => t.i);
  if (seasonChart) seasonChart.destroy();
  seasonChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "cumulative accuracy", data: timeline.map((t) => t.cum_acc),
          borderColor: "#22c55e", backgroundColor: "transparent", yAxisID: "y", pointRadius: 0, borderWidth: 2 },
        { label: "cumulative log loss", data: timeline.map((t) => t.cum_log_loss),
          borderColor: "#f5c542", backgroundColor: "transparent", yAxisID: "y1", pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true, interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#e6edf3" } },
        tooltip: {
          callbacks: {
            title: (items) => "game #" + items[0].label + " (" + timeline[items[0].dataIndex].date + ")",
          },
        },
      },
      scales: {
        x: { ticks: { color: "#8b97a7", maxTicksLimit: 12 }, grid: { color: "#222a33" }, title: { display: true, text: "game # in season", color: "#8b97a7" } },
        y: { position: "left", min: 0.4, max: 0.8, ticks: { color: "#22c55e", callback: (v) => (v * 100).toFixed(0) + "%" }, grid: { color: "#222a33" } },
        y1: { position: "right", min: 0.5, max: 0.75, ticks: { color: "#f5c542" }, grid: { drawOnChartArea: false } },
      },
    },
  });
}

init();
