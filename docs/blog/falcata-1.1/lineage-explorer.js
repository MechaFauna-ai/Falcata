/* Reader view of the existing kernel-evolution DAG: code parents, combinations and
 * idea reuse remain distinct. Archive fields are exported through an allowlist.
 * The ancestor traversal follows the existing dashboard's lineage traversal.
 */
(() => {
    const data = JSON.parse(document.getElementById("lineage-data").textContent);
    const byId = new Map(
        data.nodes.map((node) => [
            node.id,
            node,
        ]),
    );
    const campaigns = new Map(
        data.campaigns.map((campaign) => [
            campaign.id,
            campaign,
        ]),
    );
    const edges = data.edges.filter((edge) => byId.has(edge.source) && byId.has(edge.target));
    const incoming = new Map();
    for (const edge of edges) {
        if (!incoming.has(edge.target)) incoming.set(edge.target, []);
        incoming.get(edge.target).push(edge);
    }
    const state = {
        mode: "route",
        campaign: "all",
        query: "",
        outcome: "all",
        selected: null,
        ancestry: false,
        zoom: 1,
    };
    const $ = (id) => document.getElementById(id);
    const esc = (value) =>
        String(value ?? "").replace(
            /[&<>"']/g,
            (char) =>
                ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;",
                })[char],
        );
    const codeKinds = new Set([
        "parent",
        "combine",
        "integration",
    ]);
    const labels = {
        improved: "Improved",
        neutral: "Neutral",
        regressed: "Regressed",
        rejected: "Review rejected",
        refuted: "Refuted",
        failed: "Failed checks / build",
        interrupted: "Interrupted",
        noresult: "No scored result",
        pending: "Awaiting evidence",
        baseline: "Campaign baseline",
        milestone: "Integrated change",
    };
    const kind = (node) => (node.kind === "candidate" ? node.outcome || "noresult" : node.kind);
    labels.valid_unclassified = "Valid replay";
    labels.reference = "Historical reference";
    labels.reused = "Reused replay";
    labels.unknown = "Outcome unavailable";
    const label = (node) => labels[kind(node)] || kind(node);
    const short = (text, size) =>
        String(text ?? "").length > size ? `${String(text).slice(0, size - 1)}…` : String(text ?? "");
    const finite = (value) => typeof value === "number" && Number.isFinite(value);
    const noteText = (notes) => (Array.isArray(notes) ? notes.join(" ") : notes || "");
    const publicCommits = new Set(data.public_commits || (data.milestones || []).map((milestone) => milestone.commit));
    const ratio = (node) => (finite(node.ratio) ? `${node.ratio.toFixed(3)}×` : "Unscored");
    const commitLink = (commit, text = null) =>
        /^[0-9a-f]{7,40}$/.test(commit || "") && publicCommits.has(commit)
            ? `<a href="https://github.com/MechaFauna-ai/Falcata/commit/${commit}" target="_blank" rel="noreferrer">${esc(text || commit.slice(0, 10))}</a>`
            : commit
              ? `${esc(commit.slice(0, 12))} · archived search hash`
              : "No archived commit";

    function ancestors(ids, includeIdeas = false) {
        const found = new Set(ids.filter((id) => byId.has(id)));
        const up = [
            ...found,
        ];
        while (up.length) {
            for (const edge of incoming.get(up.pop()) || []) {
                if (!includeIdeas && !codeKinds.has(edge.kind)) continue;
                if (!found.has(edge.source)) {
                    found.add(edge.source);
                    up.push(edge.source);
                }
            }
        }
        return found;
    }

    const milestones = data.milestones || [];
    const highlightData = data.highlights || milestones.slice(0, 3);
    const routeSeeds = [
        ...new Set(
            milestones
                .flatMap((milestone) => [
                    ...(milestone.node_ids || []),
                    milestone.id,
                ])
                .filter((id) => byId.has(id)),
        ),
    ];
    const route = ancestors(
        routeSeeds.length ? routeSeeds : data.nodes.filter((node) => node.article_role).map((node) => node.id),
    );

    function visibleNodes() {
        let nodes = state.mode === "route" && route.size ? data.nodes.filter((node) => route.has(node.id)) : data.nodes;
        if (state.campaign !== "all") nodes = nodes.filter((node) => node.campaign === state.campaign);
        if (state.query)
            nodes = nodes.filter((node) =>
                [
                    node.id,
                    node.title,
                    node.hypothesis,
                    node.observation,
                    node.commit,
                    node.verdict_text,
                    JSON.stringify(node.hypotheses || []),
                ]
                    .join(" ")
                    .toLowerCase()
                    .includes(state.query),
            );
        if (state.outcome !== "all") nodes = nodes.filter((node) => kind(node) === state.outcome);
        if (state.ancestry && state.selected) {
            const selectedAncestry = ancestors(
                [
                    state.selected,
                ],
                true,
            );
            nodes = nodes.filter((node) => selectedAncestry.has(node.id));
        }
        // Preserve the context of every match. A filter must never silently turn
        // an ancestral node into a root or imply a different code parent.
        const matches = new Set(nodes.map((node) => node.id));
        const context = ancestors([
            ...matches,
        ]);
        return {
            nodes: data.nodes.filter((node) => context.has(node.id)),
            matches,
        };
    }

    function setSelection(id, focus = false) {
        if (!byId.has(id)) return;
        state.selected = id;
        if (state.campaign !== "all" && byId.get(id).campaign !== state.campaign) {
            state.campaign = "all";
            $("campaign").value = "all";
        }
        state.query = "";
        state.outcome = "all";
        $("search").value = "";
        $("outcome").value = "all";
        const hash = new URLSearchParams({
            candidate: id,
            view: state.mode,
        });
        if (state.campaign !== "all") hash.set("campaign", state.campaign);
        location.hash = hash.toString();
        render();
        const selected = $("graph").querySelector(`[data-node="${CSS.escape(id)}"]`);
        selected?.scrollIntoView({
            block: "center",
            inline: "center",
        });
        if (focus)
            $("detail").focus({
                preventScroll: false,
            });
    }

    function renderGraph(nodes, matches) {
        const available = new Set(nodes.map((node) => node.id));
        const links = edges.filter((edge) => available.has(edge.source) && available.has(edge.target));
        const depths = new Map();
        function depth(id, visiting = new Set()) {
            if (depths.has(id)) return depths.get(id);
            if (visiting.has(id)) return 0;
            visiting.add(id);
            const parents = links.filter((edge) => edge.target === id && codeKinds.has(edge.kind));
            const result = parents.length
                ? 1 + Math.max(...parents.map((edge) => depth(edge.source, new Set(visiting))))
                : 0;
            depths.set(id, result);
            return result;
        }
        const columns = new Map();
        for (const node of nodes) {
            const column = depth(node.id);
            if (!columns.has(column)) columns.set(column, []);
            columns.get(column).push(node);
        }
        const positions = new Map();
        const rowHeight = 81;
        const columnWidth = 226;
        const boxWidth = 198;
        let height = 0;
        for (const [column, columnNodes] of [
            ...columns,
        ].sort(([a], [b]) => a - b)) {
            const parentY = (node) => {
                const parents = links
                    .filter((edge) => edge.target === node.id && codeKinds.has(edge.kind))
                    .map((edge) => positions.get(edge.source))
                    .filter(Boolean);
                return parents.length ? parents.reduce((sum, pos) => sum + pos.y, 0) / parents.length : -1;
            };
            columnNodes.sort(
                (a, b) =>
                    parentY(a) - parentY(b) ||
                    a.id.localeCompare(b.id, undefined, {
                        numeric: true,
                    }),
            );
            // Keep each column compact after ordering by its parents. Carrying
            // the parents' absolute row positions forward leaves large empty
            // regions when several campaigns converge into one release path.
            for (const [row, node] of columnNodes.entries()) {
                const y = 8 + row * rowHeight;
                positions.set(node.id, {
                    x: column * columnWidth + 4,
                    y,
                });
                height = Math.max(height, y + 75);
            }
        }
        const selectedAncestors = state.selected
            ? ancestors(
                  [
                      state.selected,
                  ],
                  true,
              )
            : new Set();
        let paths = "";
        for (const edge of links) {
            const from = positions.get(edge.source);
            const to = positions.get(edge.target);
            const x1 = from.x + boxWidth;
            const y1 = from.y + 32;
            const x2 = to.x;
            const y2 = to.y + 32;
            const middle = (x1 + x2) / 2;
            const path =
                x2 > x1
                    ? `M${x1},${y1}C${middle},${y1} ${middle},${y2} ${x2},${y2}`
                    : `M${x1},${y1}C${x1 + 35},${y1} ${x1 + 35},${y2} ${x2 + boxWidth},${y2}`;
            const selected = selectedAncestors.has(edge.source) && selectedAncestors.has(edge.target);
            paths += `<path class="edge ${esc(edge.kind)}${selected ? " selected" : ""}" d="${path}"><title>${esc(`${edge.source} → ${edge.target}: ${edge.kind}. ${edge.evidence || ""}`)}</title></path>`;
        }
        let marks = "";
        for (const node of nodes) {
            const pos = positions.get(node.id);
            const campaign = campaigns.get(node.campaign);
            const scoreCampaign = campaigns.get(node.measurement_source?.campaign) || campaign;
            const metric = finite(node.ratio)
                ? ratio(node)
                : node.kind === "baseline"
                  ? "base"
                  : node.kind === "milestone"
                    ? "code"
                    : "—";
            marks += `<g class="node ${esc(kind(node))}${state.selected === node.id ? " selected" : ""}" transform="translate(${pos.x},${pos.y})" role="button" tabindex="0" data-node="${esc(node.id)}" aria-label="${esc(`${node.id}: ${node.title}. ${label(node)}`)}"><title>${esc(`${node.id} · ${node.title}\n${label(node)}${finite(node.ratio) ? ` · ${ratio(node)} measured within ${scoreCampaign?.label || node.campaign}` : ""}${matches.has(node.id) ? "" : " · ancestor context"}`)}</title><rect class="box" width="${boxWidth}" height="65" rx="7"/><circle cx="13" cy="16" r="3.5"/><text class="name" x="23" y="20">${esc(short(node.local_id || node.id, 16))}</text><text x="185" y="20" text-anchor="end">${esc(metric)}</text><text x="12" y="38">${esc(short(node.title, 30))}</text><text class="campaign" x="12" y="54">${esc(short(campaign?.label || node.campaign || "release", 32))}${matches.has(node.id) ? "" : " · context"}</text></g>`;
        }
        const width = (Math.max(0, ...columns.keys()) + 1) * columnWidth;
        $("graph").innerHTML = nodes.length
            ? `<svg width="${width * state.zoom}" height="${height * state.zoom}" viewBox="0 0 ${width} ${height}" aria-label="Kernel evolution lineage">${paths}${marks}</svg>`
            : '<p class="empty">No candidates match these filters. Try another campaign or search term.</p>';
        $("graph-count").textContent =
            `${matches.size} matches · ${nodes.length - matches.size} ancestor nodes · ${links.length} edges`;
        $("records-title").textContent =
            `Browse ${nodes.filter((node) => node.kind === "candidate" && matches.has(node.id)).length} matching candidates`;
    }

    function renderDetail() {
        const node = byId.get(state.selected);
        if (!node) {
            $("detail").innerHTML =
                '<p class="eyebrow">SELECT A CANDIDATE</p><h3>What was the idea?<br>Did it survive?</h3><p class="lead">Pick a highlight or a graph node to read its observation, hypothesis, test and verdict.</p><p class="muted">Solid lines track code. Violet edges track ideas. Dashed green edges identify a verified link to integrated release code.</p>';
            return;
        }
        const campaign = campaigns.get(node.campaign);
        const measurementCampaign = campaigns.get(node.measurement_source?.campaign) || campaign;
        const measurement = node.measurement_source || measurementCampaign || {};
        const sections = [
            [
                "Observation",
                node.observation,
            ],
            [
                "Hypothesis",
                node.hypothesis,
            ],
            [
                "Plan",
                node.plan,
            ],
            [
                "Falsification",
                node.falsification,
            ],
            [
                "What happened",
                node.verdict_text,
            ],
        ];
        const earlier = (node.hypotheses || [])
            .slice(0, -1)
            .map(
                (hypothesis, index) =>
                    `<details><summary>Earlier hypothesis ${index + 1}: ${esc(hypothesis.hyp_verdict || "verdict unavailable")}</summary>${[
                        "observation",
                        "hypothesis",
                        "plan",
                        "falsification",
                        "verdict_text",
                    ]
                        .filter((key) => hypothesis[key])
                        .map((key) => `<h4>${esc(key.replaceAll("_", " "))}</h4><p>${esc(hypothesis[key])}</p>`)
                        .join("")}</details>`,
            )
            .join("");
        const badges = `<span class="badge ${esc(kind(node))}">${esc(label(node))}</span>${node.hyp_verdict ? `<span class="badge">Latest hypothesis: ${esc(node.hyp_verdict)}</span>` : ""}${node.archive_hyp_verdict && node.archive_hyp_verdict !== node.hyp_verdict ? `<span class="badge">Archive summary: ${esc(node.archive_hyp_verdict)}</span>` : ""}${node.accepted ? '<span class="badge">Accepted in search</span>' : ""}`;
        let measurementHtml = "";
        if (finite(node.ratio)) {
            measurementHtml = `<div class="measurement"><strong>${ratio(node)}</strong><p>${esc(measurement.metric || "Archived ratio vs campaign baseline")}</p>${Array.isArray(node.ci) && node.ci.length === 2 ? `<p>95% interval ${node.ci.map((value) => (finite(value) ? value.toFixed(3) : "unknown")).join("–")}×</p>` : ""}<p class="muted">${esc(measurementCampaign?.workload || "")}${measurement.rounds ? ` · ${esc(measurement.rounds)} requested rounds` : ""}${measurement.pairs ? ` · ${esc(measurement.pairs)} configured pairs` : ""}</p>${node.measurement_note ? `<p>${esc(node.measurement_note)}</p>` : ""}<p class="muted">Baseline ${esc((measurement.base_commit || measurementCampaign?.base_commit || "unknown").slice(0, 12))}. Search evidence; compare only within this protocol. ${esc(noteText(measurementCampaign?.notes))}</p></div>`;
        } else if (finite(node.quick_ratio)) {
            measurementHtml = `<div class="measurement"><p>Worker quick check: ${node.quick_ratio.toFixed(3)}×</p><p class="muted">A diagnostic proxy. No scored operator result is archived for this candidate.</p></div>`;
        } else if (node.kind === "candidate")
            measurementHtml =
                '<div class="measurement"><p>No scored timing result.</p><p class="muted">This attempt remains in the lineage.</p></div>';
        const checks = Object.entries(node.checks || {})
            .filter(([, value]) => value !== null && value !== undefined)
            .map(
                ([key, value]) =>
                    `<p>${esc(key.replaceAll("_", " "))}: ${esc(typeof value === "object" ? JSON.stringify(value) : value)}</p>`,
            )
            .join("");
        const relations = edges
            .filter((edge) => edge.target === node.id || edge.source === node.id)
            .map((edge) => {
                const other = edge.target === node.id ? edge.source : edge.target;
                return `<button type="button" data-select="${esc(other)}" title="${esc(edge.evidence || "")}">${edge.target === node.id ? "←" : "→"} ${esc(byId.get(other).local_id || other)} · ${esc(edge.kind)}</button>`;
            })
            .join("");
        const full = node.full_bench || {};
        const fullEvidence = finite(full.trees_per_s)
            ? `<h4>Separate full benchmark</h4><p>${full.trees_per_s.toFixed(1)} trees/s${finite(full.train_s) ? ` · ${full.train_s.toFixed(2)} s training` : ""}${full.rounds ? ` · ${esc(full.rounds)} requested rounds` : ""}</p><p class="muted">A separate measurement. Refer to the benchmark companion for timing and quality scope.</p>`
            : "";
        $("detail").innerHTML =
            `<div class="detail-id"><span>${esc(node.id)}</span></div><p class="muted">${esc(campaign?.label || "release")}</p><h3>${esc(short(node.title, 100))}</h3>${node.title.length > 100 ? `<details><summary>Full implementation title</summary><p>${esc(node.title)}</p></details>` : ""}<div>${badges}</div>${measurementHtml}${sections
                .filter(([, value]) => value)
                .map(([heading, value]) => `<h4>${heading}</h4><p>${esc(value)}</p>`)
                .join(
                    "",
                )}${earlier}${node.reason ? `<h4>Recorded status</h4><p>${esc(node.reason)}</p>` : ""}${checks ? `<details><summary>Checks and review</summary>${checks}</details>` : ""}${fullEvidence}<h4>Code</h4><p>${commitLink(node.commit)}</p>${node.parent_commit ? `<p class="muted">Recorded code parent: ${commitLink(node.parent_commit)}</p>` : ""}${node.source_hash ? `<p class="muted">Archived source SHA256: ${esc(node.source_hash)}</p>` : ""}${node.evidence ? `<h4>Archive reference</h4><p>${esc(node.evidence)}</p>` : ""}${relations ? `<h4>Connected experiments</h4><div class="relations">${relations}</div>` : ""}`;
    }

    function render() {
        const { nodes, matches } = visibleNodes();
        renderGraph(nodes, matches);
        renderDetail();
        $("candidate-list").innerHTML = nodes
            .filter((node) => node.kind === "candidate" && matches.has(node.id))
            .map(
                (node) =>
                    `<div class="candidate-list"><button type="button" data-select="${esc(node.id)}">${esc(node.id)}</button><small>${esc(label(node))}</small><span>${esc(node.title)}</span><span>${esc(ratio(node))}</span></div>`,
            )
            .join("");
        $("route-view").setAttribute("aria-pressed", String(state.mode === "route"));
        $("all-view").setAttribute("aria-pressed", String(state.mode === "all"));
        $("scope").textContent =
            state.campaign === "all"
                ? "Ratios use different campaign baselines. Select a campaign or candidate for the metric and protocol; no cross-campaign ranking is implied."
                : `${noteText(campaigns.get(state.campaign)?.notes)} Ancestors from other campaigns remain visible as context.`;
    }

    function selectFromEvent(event) {
        const control = event.target.closest("[data-select], [data-node]");
        if (control) setSelection(control.dataset.select || control.dataset.node, window.innerWidth < 761);
    }
    function clearAncestryFocus() {
        state.ancestry = false;
        $("ancestry").checked = false;
    }
    $("graph").addEventListener("click", selectFromEvent);
    $("graph").addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectFromEvent(event);
        }
    });
    $("detail").addEventListener("click", selectFromEvent);
    $("candidate-list").addEventListener("click", selectFromEvent);
    $("campaign").addEventListener("change", (event) => {
        clearAncestryFocus();
        state.campaign = event.target.value;
        state.mode = "all";
        render();
    });
    $("outcome").addEventListener("change", (event) => {
        clearAncestryFocus();
        state.outcome = event.target.value;
        state.mode = "all";
        render();
    });
    $("search").addEventListener("input", (event) => {
        clearAncestryFocus();
        state.query = event.target.value.trim().toLowerCase();
        state.mode = "all";
        render();
    });
    $("ancestry").addEventListener("change", (event) => {
        state.ancestry = event.target.checked;
        render();
    });
    $("route-view").addEventListener("click", () => {
        clearAncestryFocus();
        state.mode = "route";
        state.campaign = "all";
        state.outcome = "all";
        state.query = "";
        $("campaign").value = "all";
        $("outcome").value = "all";
        $("search").value = "";
        render();
    });
    $("all-view").addEventListener("click", () => {
        clearAncestryFocus();
        state.mode = "all";
        render();
    });
    $("zoom-in").addEventListener("click", () => {
        state.zoom = Math.min(1.8, state.zoom + 0.15);
        render();
    });
    $("zoom-out").addEventListener("click", () => {
        state.zoom = Math.max(0.4, state.zoom - 0.15);
        render();
    });
    $("reset").addEventListener("click", () => {
        Object.assign(state, {
            mode: "route",
            campaign: "all",
            query: "",
            outcome: "all",
            selected: null,
            ancestry: false,
            zoom: 1,
        });
        $("campaign").value = "all";
        $("outcome").value = "all";
        $("search").value = "";
        $("ancestry").checked = false;
        location.hash = "";
        render();
        $("graph-scroll").scrollTo(0, 0);
    });
    $("theme").addEventListener("click", () => {
        const dark = document.documentElement.dataset.theme !== "dark";
        document.documentElement.dataset.theme = dark ? "dark" : "light";
        $("theme").textContent = dark ? "Light theme" : "Dark theme";
        $("theme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    });
    for (const campaign of data.campaigns)
        $("campaign").insertAdjacentHTML(
            "beforeend",
            `<option value="${esc(campaign.id)}">${esc(campaign.label)}</option>`,
        );
    const candidates = data.nodes.filter((node) => node.kind === "candidate");
    const stats = [
        [
            candidates.length,
            "candidate / replay records",
        ],
        [
            data.campaigns.length,
            "search campaigns",
        ],
        [
            candidates.filter((node) => node.accepted).length,
            "accepted in search",
        ],
        [
            candidates.filter((node) => !finite(node.ratio)).length,
            "without scored ratios",
        ],
    ];
    $("stats").innerHTML = stats
        .map(([value, text]) => `<div class="stat"><strong>${value}</strong><span>${text}</span></div>`)
        .join("");
    $("highlights").innerHTML = highlightData
        .slice(0, 3)
        .map(
            (highlight, index) =>
                `<article class="highlight"><span class="number">0${index + 1} / MECHANISM</span><h3>${esc(highlight.title)}</h3><p>${esc(highlight.summary)}</p><button type="button" data-highlight="${index}">Follow this idea ↗</button></article>`,
        )
        .join("");
    $("highlights").addEventListener("click", (event) => {
        const button = event.target.closest("[data-highlight]");
        if (!button) return;
        const highlight = highlightData[Number(button.dataset.highlight)];
        const id = highlight.focus_id || highlight.node_ids?.find((candidate) => byId.has(candidate));
        if (!id) return;
        state.mode = "all";
        state.campaign = "all";
        state.outcome = "all";
        state.query = "";
        state.ancestry = true;
        $("campaign").value = "all";
        $("outcome").value = "all";
        $("search").value = "";
        $("ancestry").checked = true;
        setSelection(id, window.innerWidth < 761);
        $("explorer-title").scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
    });
    $("limits").innerHTML = (data.limitations || []).map((limit) => `<li>${esc(limit)}</li>`).join("");
    const hash = new URLSearchParams(location.hash.slice(1));
    if (byId.has(hash.get("candidate"))) {
        state.selected = hash.get("candidate");
        state.mode = hash.get("view") === "route" ? "route" : "all";
    }
    if (campaigns.has(hash.get("campaign"))) {
        state.campaign = hash.get("campaign");
        $("campaign").value = state.campaign;
    }
    render();
})();
