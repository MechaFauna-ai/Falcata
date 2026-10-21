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
    // Campaigns in chronological order (start date, then end date); the export keeps source order.
    const campaignList = data.campaigns
        .map((campaign, index) => ({
            campaign,
            index,
        }))
        .sort(
            (a, b) =>
                String(a.campaign.start || "9999").localeCompare(String(b.campaign.start || "9999")) ||
                String(a.campaign.end || "9999").localeCompare(String(b.campaign.end || "9999")) ||
                a.index - b.index,
        )
        .map(({ campaign }) => campaign);
    const campaigns = new Map(
        campaignList.map((campaign) => [
            campaign.id,
            campaign,
        ]),
    );
    // Milestones and unresolved references belong to no search campaign; they are browsable as
    // pseudo-campaigns so deep links to them keep a valid campaign selection.
    const pseudoCampaigns = new Map(
        [
            [
                "release",
                {
                    id: "release",
                    label: "Release integrations",
                    notes: [
                        "Integrated changes on Falcata master, each linked to its commit and pull request.",
                    ],
                },
            ],
            [
                "unknown",
                {
                    id: "unknown",
                    label: "Unresolved references",
                    notes: [
                        "Candidates named by a record whose own record is unavailable.",
                    ],
                },
            ],
        ].filter(([id]) => data.nodes.some((node) => node.campaign === id)),
    );
    const campaignInfo = (id) => campaigns.get(id) || pseudoCampaigns.get(id);
    const campaignLabel = (id) => campaignInfo(id)?.label || id;
    const campaignOrder = new Map(
        [
            ...campaignList.map((campaign) => campaign.id),
            ...pseudoCampaigns.keys(),
        ].map((id, index) => [
            id,
            index,
        ]),
    );
    const months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ];
    const dayMonth = (iso) => {
        const [year, month, day] = String(iso).split("-").map(Number);
        return {
            year,
            month: months[month - 1] || "",
            day,
        };
    };
    function dateRange(campaign) {
        if (!campaign?.start) return "";
        const a = dayMonth(campaign.start);
        const b = dayMonth(campaign.end || campaign.start);
        if (campaign.start === (campaign.end || campaign.start)) return `${a.day} ${a.month} ${a.year}`;
        if (a.year === b.year && a.month === b.month) return `${a.day}–${b.day} ${b.month} ${b.year}`;
        if (a.year === b.year) return `${a.day} ${a.month} – ${b.day} ${b.month} ${b.year}`;
        return `${a.day} ${a.month} ${a.year} – ${b.day} ${b.month} ${b.year}`;
    }
    const edges = data.edges.filter((edge) => byId.has(edge.source) && byId.has(edge.target));
    const incoming = new Map();
    for (const edge of edges) {
        if (!incoming.has(edge.target)) incoming.set(edge.target, []);
        incoming.get(edge.target).push(edge);
    }
    const state = {
        mode: "all",
        campaign: "all",
        query: "",
        outcome: "all",
        speed: "all",
        checks: "all",
        grouped: true,
        page: 0,
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
        improved: "Improved vs parent",
        neutral: "Neutral vs parent",
        regressed: "Regressed vs parent",
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
    const firstLine = (text) =>
        String(text ?? "")
            .split("\n")
            .map((line) => line.replace(/^[\s>*#-]+/, "").trim())
            .find(Boolean) || "";
    // Records whose worker wrote no summary are titled by their first hypothesis line.
    function displayTitle(node) {
        if (node.title && node.title !== "(no summary)") return node.title;
        const hypothesis =
            firstLine(node.hypothesis) ||
            firstLine(node.archive_hypothesis_text) ||
            firstLine((node.hypotheses || []).map((entry) => entry.hypothesis).find(Boolean));
        return hypothesis ? `No summary · ${hypothesis}` : node.title || node.id;
    }
    const noteText = (notes) => (Array.isArray(notes) ? notes.join(" ") : notes || "");
    const publicCommits = new Set(data.public_commits || (data.milestones || []).map((milestone) => milestone.commit));
    const ratio = (node) => (finite(node.ratio) ? `${node.ratio.toFixed(3)}×` : "Unscored");
    const candidates = data.nodes
        .map((node, index) => ({
            node,
            index,
        }))
        .filter(({ node }) => node.kind === "candidate")
        .sort(
            (a, b) =>
                (campaignOrder.get(a.node.campaign) ?? 1e9) - (campaignOrder.get(b.node.campaign) ?? 1e9) ||
                a.index - b.index,
        )
        .map(({ node }) => node);
    const pageSize = 12;
    const graphCap = 150;
    const isReplay = (node) => Boolean(node.source_hash);
    const replayCampaign = (id) => candidates.some((node) => node.campaign === id && isReplay(node));
    const speedClass = (node) => (!finite(node.ratio) ? "unscored" : node.ratio > 1 ? "faster" : "slower");
    function assessment(node) {
        const checks = node.checks || {};
        const reasons = [];
        if (checks.later_expanded_coverage_passed === false) reasons.push("Later coverage failed");
        if (checks.review === "reject" || node.outcome === "rejected") reasons.push("Review rejected");
        if (checks.byte_exact_valid === false) reasons.push("Byte-exact check failed");
        if (checks.md5_ok === false) reasons.push("Identity gate not passed");
        if (checks.lattice_ok !== null && checks.lattice_ok !== undefined && checks.lattice_ok !== "ok")
            reasons.push("Broader checks failed");
        if (checks.suspect === true) reasons.push("Timing marked suspect");
        if (node.outcome === "failed" && !reasons.length) reasons.push("Archived checks / build failure");
        if (reasons.length)
            return {
                status: "blocked",
                reasons,
            };
        const passed =
            (node.status === "valid" && checks.byte_exact_valid === true) ||
            ([
                "ok",
                "regressed",
            ].includes(node.status) &&
                checks.md5_ok === true &&
                checks.lattice_ok === "ok");
        return {
            status: passed ? "passed" : "unknown",
            reasons: [],
        };
    }
    function presentationKind(node) {
        if (node.kind !== "candidate") return node.kind;
        if (assessment(node).status === "blocked") return "failed";
        if (speedClass(node) === "unscored") return "noresult";
        return speedClass(node) === "faster" && assessment(node).status === "passed" ? "improved" : "neutral";
    }
    const cached = (node) => node.outcome === "reused" || node.checks?.duplicate_reused === true;
    const metricLabel = (node) => (isReplay(node) ? "vs kernel baseline" : "vs measurement baseline");
    function checkLabel(node) {
        const result = assessment(node);
        return result.status === "blocked"
            ? result.reasons.join(" · ")
            : result.status === "passed"
              ? "Recorded checks passed"
              : "Checks not established";
    }
    const textKeys = [
        "observation",
        "hypothesis",
        "plan",
        "falsification",
        "verdict_text",
    ];
    const searchCache = new Map();
    // Search covers the record's text and identifiers only, never serialized field names.
    function searchText(node) {
        if (!searchCache.has(node.id))
            searchCache.set(
                node.id,
                [
                    node.id,
                    node.title,
                    displayTitle(node),
                    node.commit,
                    node.source_hash,
                    node.archive_hypothesis_text,
                    ...textKeys.map((key) => node[key]),
                    ...(node.hypotheses || []).flatMap((entry) => textKeys.map((key) => entry[key])),
                    node.registered_idea?.mechanism,
                    node.registered_idea?.family,
                ]
                    .filter((value) => typeof value === "string" && value)
                    .join("\n")
                    .toLowerCase(),
            );
        return searchCache.get(node.id);
    }
    function matchesFilters(node) {
        return (
            (state.campaign === "all" || node.campaign === state.campaign) &&
            (state.outcome === "all" || kind(node) === state.outcome) &&
            (state.speed === "all" || speedClass(node) === state.speed) &&
            (state.checks === "all" || assessment(node).status === state.checks) &&
            (!state.query || searchText(node).includes(state.query))
        );
    }
    const groupKey = (node) =>
        isReplay(node)
            ? JSON.stringify([
                  node.campaign,
                  node.source_hash,
                  node.measurement_source || {
                      baseline: campaigns.get(node.campaign)?.base_commit,
                      metric: campaigns.get(node.campaign)?.metric,
                  },
              ])
            : node.id;
    const allGroups = new Map();
    for (const node of candidates) {
        const key = groupKey(node);
        if (!allGroups.has(key)) allGroups.set(key, []);
        allGroups.get(key).push(node);
    }
    function matchingRecords() {
        return candidates.filter((node) => matchesFilters(node) && (state.mode !== "route" || route.has(node.id)));
    }
    function matchingGroups(records) {
        const groups = new Map();
        for (const node of records) {
            const key = state.grouped ? groupKey(node) : node.id;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(node);
        }
        return [
            ...groups.values(),
        ];
    }
    function syncControls() {
        for (const [id, value] of [
            [
                "campaign",
                state.campaign,
            ],
            [
                "search",
                state.query,
            ],
            [
                "outcome",
                state.outcome,
            ],
            [
                "speed",
                state.speed,
            ],
            [
                "check-filter",
                state.checks,
            ],
        ])
            $(id).value = value;
        $("ancestry").checked = state.ancestry;
        $("group-replays").checked = state.grouped;
    }
    function saveHash() {
        const hash = new URLSearchParams({
            view: state.mode,
        });
        if (state.selected) hash.set("candidate", state.selected);
        hash.set("campaign", state.campaign);
        for (const key of [
            "query",
            "outcome",
            "speed",
            "checks",
        ])
            if (state[key] && state[key] !== "all") hash.set(key, state[key]);
        if (!state.grouped) hash.set("grouped", "false");
        if (state.page) hash.set("page", String(state.page));
        if (state.ancestry) hash.set("ancestry", "true");
        history.replaceState(null, "", `#${hash}`);
    }
    const prLink = (node) =>
        Number.isInteger(node.pr) &&
        /^https:\/\/github\.com\/MechaFauna-ai\/Falcata\/pull\/\d+$/.test(node.pr_url || "")
            ? `<a href="${esc(node.pr_url)}" target="_blank" rel="noreferrer">Pull request #${node.pr}</a>`
            : "";
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
        const scoped = data.nodes.filter((node) =>
            state.mode === "route" ? route.has(node.id) : node.campaign === state.campaign,
        );
        let nodes = scoped.filter(matchesFilters);
        if (state.ancestry && state.selected) {
            const selectedAncestry = ancestors(
                [
                    state.selected,
                ],
                true,
            );
            nodes = nodes.filter((node) => selectedAncestry.has(node.id));
        }
        const matches = new Set(nodes.map((node) => node.id));
        const allowed = new Set(scoped.map((node) => node.id));
        const context = new Set(matches);
        const pending = [
            ...matches,
        ];
        while (pending.length)
            for (const edge of incoming.get(pending.pop()) || []) {
                if (codeKinds.has(edge.kind) && allowed.has(edge.source) && !context.has(edge.source)) {
                    context.add(edge.source);
                    pending.push(edge.source);
                }
            }
        // Local context stays inside this campaign. Immediate external parents
        // are marked boundary references instead of recursively opening history.
        nodes = scoped.filter((node) => context.has(node.id));
        // The release view holds every milestone with its recorded code ancestry (93 nodes for
        // the 21 October export); the largest single campaign view is about 30. The cap only guards
        // against an unbounded SVG.
        const omitted = Math.max(0, nodes.length - graphCap);
        nodes = nodes.slice(0, graphCap);
        const external = new Set();
        for (const node of nodes)
            for (const edge of incoming.get(node.id) || []) {
                if (codeKinds.has(edge.kind) && !context.has(edge.source)) external.add(edge.source);
            }
        const references = [
            ...external,
        ]
            .slice(0, 12)
            .map((id) => ({
                ...byId.get(id),
                boundary: true,
                kind: "reference",
                ratio: null,
                title: `Outside view: ${displayTitle(byId.get(id))}`,
            }));
        return {
            nodes: [
                ...references,
                ...nodes,
            ],
            matches,
            omitted: omitted + Math.max(0, external.size - 12),
        };
    }

    function setSelection(id, focus = false) {
        if (!byId.has(id)) return;
        // Re-rendering replaces the graph and lists; remember where keyboard focus was.
        const active = document.activeElement;
        const inGraph = $("graph").contains(active);
        const inList = $("candidate-list").contains(active);
        state.selected = id;
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
        else if (inGraph && selected)
            selected.focus({
                preventScroll: true,
            });
        else if (inList)
            $("candidate-list")
                .querySelector(`[data-select="${CSS.escape(id)}"]`)
                ?.focus({
                    preventScroll: true,
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
            const metric = node.boundary
                ? "external"
                : finite(node.ratio)
                  ? `${ratio(node)} / base`
                  : node.kind === "baseline"
                    ? "base"
                    : node.kind === "milestone"
                      ? "code"
                      : "—";
            marks += `<g class="node ${esc(presentationKind(node))}${node.boundary ? " boundary" : ""}${state.selected === node.id ? " selected" : ""}" transform="translate(${pos.x},${pos.y})" role="button" tabindex="0" data-node="${esc(node.id)}" aria-label="${esc(`${node.id}: ${displayTitle(node)}. ${label(node)}. ${checkLabel(node)}`)}"><title>${esc(`${node.id} · ${displayTitle(node)}\n${label(node)} · ${checkLabel(node)}${finite(node.ratio) ? ` · ${ratio(node)} ${metricLabel(node)} within ${scoreCampaign?.label || node.campaign}` : ""}${matches.has(node.id) ? "" : " · context"}`)}</title><rect class="box" width="${boxWidth}" height="65" rx="7"/><circle cx="13" cy="16" r="3.5"/><text class="name" x="23" y="20">${esc(short(node.local_id || node.id, 13))}</text><text x="185" y="20" text-anchor="end">${esc(metric)}</text><text x="12" y="38">${esc(short(node.boundary ? node.title : displayTitle(node), 30))}</text><text class="campaign" x="12" y="55">${esc(node.boundary ? `Outside · ${short(campaignLabel(node.campaign), 20)}` : short(checkLabel(node), 28))}${!node.boundary && !matches.has(node.id) ? " · context" : ""}</text></g>`;
        }
        const width = (Math.max(0, ...columns.keys()) + 1) * columnWidth;
        $("graph").innerHTML = nodes.length
            ? `<svg width="${width * state.zoom}" height="${height * state.zoom}" viewBox="0 0 ${width} ${height}" aria-label="Kernel evolution lineage">${paths}${marks}</svg>`
            : '<p class="empty">No candidates match these filters. Try another campaign or search term.</p>';
        $("graph-count").textContent =
            `${matches.size} matches · ${nodes.filter((node) => node.boundary).length} outside references · ${links.length} edges`;
    }

    const milestoneById = new Map(
        (data.milestones || []).map((milestone) => [
            milestone.id,
            milestone,
        ]),
    );
    function milestoneSummary(node) {
        const milestone = milestoneById.get(node.id);
        return milestone?.summary ? `<h4>Integration evidence</h4><p>${esc(milestone.summary)}</p>` : "";
    }
    function renderDetail() {
        const node = byId.get(state.selected);
        if (!node) {
            $("detail").innerHTML =
                '<p class="eyebrow">EXPLORE A BRANCH</p><h3>What changed?<br>What happened next?</h3><p class="lead">Open a campaign and select a candidate, or follow the release lineage. Its detail shows the hypothesis, test, verdict and connected experiments.</p><p class="muted">Solid lines track code inheritance. Violet edges track ideas. Dashed green edges identify a verified link to integrated release code.</p>';
            return;
        }
        const campaign = campaignInfo(node.campaign);
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
        const badges = `<span class="badge ${esc(presentationKind(node))}">${esc(checkLabel(node))}</span>${label(node) !== checkLabel(node) ? `<span class="badge">${esc(label(node))}</span>` : ""}${cached(node) ? '<span class="badge cached">Cached / reused result</span>' : ""}${node.hyp_verdict ? `<span class="badge">Latest hypothesis: ${esc(node.hyp_verdict)}</span>` : ""}${node.archive_hyp_verdict && node.archive_hyp_verdict !== node.hyp_verdict ? `<span class="badge">Archive summary: ${esc(node.archive_hyp_verdict)}</span>` : ""}${node.accepted ? '<span class="badge">Accepted in search</span>' : ""}`;
        const integrated =
            (incoming.get(node.id) || []).some((edge) => edge.kind === "integration") ||
            edges.some((edge) => edge.source === node.id && edge.kind === "integration");
        const interpretation =
            node.kind === "candidate"
                ? `<div class="interpretation"><p>${
                      finite(node.ratio)
                          ? `${ratio(node)} ${metricLabel(node)} is ${node.ratio > 1 ? "a faster point estimate" : "at or below baseline speed"}. ${Array.isArray(node.ci) && node.ci[0] <= 1 && node.ci[1] >= 1 ? "Its interval includes 1×." : ""}`
                          : "No scored timing supports a speed claim for this record."
                  }</p><p>${esc(checkLabel(node))}. ${cached(node) ? "This is reused evidence, not a fresh timing draw. " : ""}${isReplay(node) ? "An isolated kernel result does not measure full training speed." : "The search decision and review are separate from the baseline ratio."}</p><p>${integrated ? "A verified integration connection is recorded below." : "No release integration connection is recorded for this candidate."}${!isReplay(node) && !node.checks?.review ? " Independent review is unrecorded." : ""}</p></div>`
                : "";
        const members = allGroups.get(groupKey(node)) || [];
        const groupDetail =
            members.length > 1
                ? `<div class="group-detail"><label>Original records in this source group<select id="group-member">${members.map((member) => `<option value="${esc(member.id)}"${member.id === node.id ? " selected" : ""}>${esc(`${short(member.id, 48)} · ${ratio(member)} · ${label(member)}${cached(member) ? " · cached" : ""}`)}</option>`).join("")}</select></label><p>${members.length} original records, including those outside the current filters. No best-only selection.</p></div>`
                : "";
        const outside = !matchesFilters(node)
            ? '<p class="selection-note">Selected record is outside the current filters. Your filters are preserved.</p>'
            : "";
        const openCampaign =
            campaign && (state.campaign !== node.campaign || state.mode === "route")
                ? `<button type="button" data-campaign="${esc(node.campaign)}">Open this campaign ↗</button>`
                : "";
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
            `<div class="detail-id"><span>${esc(node.id)}</span></div><p class="muted">${esc(campaignLabel(node.campaign))}${dateRange(campaigns.get(node.campaign)) ? ` · ${esc(dateRange(campaigns.get(node.campaign)))}` : ""}</p>${openCampaign}${outside}<h3>${esc(short(displayTitle(node), 100))}</h3>${displayTitle(node).length > 100 ? `<details><summary>Full implementation title</summary><p>${esc(displayTitle(node))}</p></details>` : ""}<div>${badges}</div>${interpretation}${measurementHtml}${groupDetail}${sections
                .filter(([, value]) => value)
                .map(([heading, value]) => `<h4>${heading}</h4><p>${esc(value)}</p>`)
                .join(
                    "",
                )}${earlier}${node.reason ? `<h4>Recorded status</h4><p>${esc(node.reason)}</p>` : ""}${checks ? `<details><summary>Checks and review</summary>${checks}</details>` : ""}${fullEvidence}<h4>Code</h4><p>${commitLink(node.commit)}${prLink(node) ? ` · ${prLink(node)}` : ""}</p>${node.kind === "milestone" ? milestoneSummary(node) : ""}${
                node.registered_idea?.mechanism
                    ? `<h4>Registered idea</h4><p>${esc(node.registered_idea.mechanism)}</p><p class="muted">${esc(
                          [
                              node.registered_idea.family,
                              node.registered_idea.phase_kernel,
                          ]
                              .filter(Boolean)
                              .join(" · "),
                      )}</p>`
                    : ""
            }${node.parent_commit ? `<p class="muted">Recorded code parent: ${commitLink(node.parent_commit)}</p>` : ""}${node.source_hash ? `<p class="muted">Archived source SHA256: ${esc(node.source_hash)}</p>` : ""}${node.evidence ? `<h4>Archive reference</h4><p>${esc(node.evidence)}</p>` : ""}${relations ? `<h4>Connected experiments</h4><div class="relations">${relations}</div>` : ""}`;
    }

    function renderHighlights() {
        const items = (data.highlights || []).filter((highlight) => byId.has(highlight.focus_id));
        if (!items.length) return "";
        return `<div class="highlights"><p class="eyebrow">START HERE</p><div class="highlight-grid">${items
            .map(
                (highlight) =>
                    `<article class="highlight-card"><h3>${esc(highlight.title)}</h3><p>${esc(highlight.summary)}</p><div class="highlight-nodes">${(
                        highlight.node_ids || []
                    )
                        .filter((id) => byId.has(id))
                        .map(
                            (id) =>
                                `<button type="button" data-select="${esc(id)}" title="${esc(displayTitle(byId.get(id)))}">${esc(byId.get(id).kind === "milestone" ? short(byId.get(id).title, 28) : id)}</button>`,
                        )
                        .join(
                            "",
                        )}</div><button type="button" class="highlight-open" data-highlight="${esc(highlight.focus_id)}">Follow ${esc(byId.get(highlight.focus_id).local_id || highlight.focus_id)} in its campaign ↗</button></article>`,
            )
            .join("")}</div></div>`;
    }
    function renderOverview(records) {
        const cards = campaignList
            .map((campaign) => {
                const local = records.filter((node) => node.campaign === campaign.id);
                const total = candidates.filter((node) => node.campaign === campaign.id);
                const faster = local.filter(
                    (node) => speedClass(node) === "faster" && assessment(node).status === "passed",
                ).length;
                const blocked = local.filter((node) => assessment(node).status === "blocked").length;
                const unscored = local.filter((node) => !finite(node.ratio)).length;
                return `<article class="campaign-card${local.length ? "" : " no-matches"}"><div class="campaign-kicker">${replayCampaign(campaign.id) ? "ISOLATED KERNEL REPLAY" : "LIBRARY SEARCH"}</div><h3>${esc(campaign.label)}</h3>${dateRange(campaign) ? `<p class="campaign-dates">${esc(dateRange(campaign))}</p>` : ""}<p>${local.length} / ${total.length} matching records${replayCampaign(campaign.id) ? ` · ${matchingGroups(local).length} source groups` : ""}</p><div class="campaign-counts"><span>${faster} faster + checks passed</span><span>${blocked} failed / rejected</span><span>${unscored} unscored</span></div><button type="button" data-campaign="${esc(campaign.id)}">Open campaign ↗</button></article>`;
            })
            .join("");
        $("graph").innerHTML = `${renderHighlights()}<div class="campaign-grid">${cards}</div>`;
        $("graph-count").textContent = `${data.campaigns.length} campaigns · ${records.length} matching records`;
        $("graph-help").textContent =
            "Campaigns have different baselines and timing protocols. Open one to explore its local lineage or replay sources. Counts are not a cross-campaign speed ranking.";
    }
    function renderReplay(records) {
        const campaign = campaigns.get(state.campaign);
        const groups = matchingGroups(records);
        $("graph").innerHTML =
            `<div class="replay-summary"><p class="eyebrow">ISOLATED KERNEL REPLAY</p><h3>${esc(campaign.label)}</h3><p>${records.length} matching records in ${groups.length} ${state.grouped ? "source groups" : "individual entries"}. These replay records have no archived code-parent edges; the grouped list below preserves every original result.</p><p>${esc(campaign.metric)}. ${esc(campaign.workload)}</p><p class="muted">Exact source hashes are grouped only within the same campaign and measurement protocol. Score ranges show the recorded spread, not a selected best or a new average. Reused results are identified individually.</p><a href="#records-panel">Browse replay sources ↓</a></div>`;
        $("graph-count").textContent = `${groups.length} entries · ${records.length} matching records`;
        $("graph-help").textContent =
            "Select a source group below, then use its original-record selector to inspect fresh, cached and failed records separately.";
    }
    function renderRecords(records) {
        const groups = matchingGroups(records);
        const pageCount = Math.max(1, Math.ceil(groups.length / pageSize));
        state.page = Math.min(state.page, pageCount - 1);
        $("candidate-list").innerHTML =
            groups
                .slice(state.page * pageSize, (state.page + 1) * pageSize)
                .map((matching) => {
                    const first =
                        matching.find((node) => !cached(node) && assessment(node).status === "passed") || matching[0];
                    const members = state.grouped ? allGroups.get(groupKey(first)) : matching;
                    const scores = members.filter((node) => finite(node.ratio)).map((node) => node.ratio);
                    const min = Math.min(...scores),
                        max = Math.max(...scores);
                    const score = !scores.length
                        ? "Unscored"
                        : min === max
                          ? `${min.toFixed(3)}×`
                          : `${min.toFixed(3)}–${max.toFixed(3)}×`;
                    const blocked = members.filter((node) => assessment(node).status === "blocked").length;
                    const reused = members.filter(cached).length;
                    return `<article class="result-row${members.some((node) => node.id === state.selected) ? " selected" : ""}"><div class="result-name"><small>${esc(campaignLabel(first.campaign))}</small><button type="button" data-select="${esc(first.id)}">${esc(short(displayTitle(first), 100))}</button><span class="muted result-id">${esc(isReplay(first) && state.grouped ? `Source ${first.source_hash.slice(0, 12)}` : first.id)}</span></div><div class="result-status"><span class="badge ${esc(presentationKind(first))}">${esc(checkLabel(first))}</span>${!isReplay(first) && label(first) !== checkLabel(first) ? `<span class="badge">${esc(label(first))}</span>` : ""}${blocked && members.length > 1 ? `<span class="badge failed">${blocked} failed / rejected record${blocked === 1 ? "" : "s"}</span>` : ""}${reused ? `<span class="badge cached">${reused} cached / reused</span>` : ""}<p>${members.length > 1 ? `${matching.length} matching / ${members.length} original records` : "1 original record"}</p></div><div class="result-score"><strong>${score}</strong><span>${esc(metricLabel(first))}</span>${members.length > 1 && scores.length ? "<small>All original scores in this group</small>" : ""}</div></article>`;
                })
                .join("") || '<p class="empty">No records match these filters.</p>';
        $("records-title").textContent =
            `Browse ${groups.length} ${state.grouped ? "grouped entries" : "individual entries"} · ${records.length} matching records`;
        $("page-status").textContent = groups.length
            ? `Page ${state.page + 1} of ${pageCount} · entries ${state.page * pageSize + 1}–${Math.min((state.page + 1) * pageSize, groups.length)}`
            : "No matching entries";
        $("page-prev").disabled = state.page === 0;
        $("page-next").disabled = state.page >= pageCount - 1;
    }
    function render() {
        const records = matchingRecords();
        const overview = state.mode === "all" && state.campaign === "all";
        const replay = state.mode === "all" && replayCampaign(state.campaign);
        $("workspace").className = `workspace${overview ? " overview" : replay ? " replay" : ""}`;
        $("ancestry-control").hidden = overview || replay;
        $("zoom-in").hidden = overview || replay;
        $("zoom-out").hidden = overview || replay;
        $("legend").hidden = overview || replay;
        if (overview) renderOverview(records);
        else if (replay) renderReplay(records);
        else {
            const { nodes, matches, omitted } = visibleNodes();
            renderGraph(nodes, matches);
            $("graph-help").textContent =
                `Only local ancestor context is expanded. Dashed boundary nodes mark parents outside this view; their complete relationships remain in the detail panel.${omitted ? ` ${omitted} additional context nodes are omitted from this bounded graph.` : ""}`;
        }
        renderDetail();
        renderRecords(records);
        if (replay || state.query || state.speed !== "all" || state.checks !== "all" || state.outcome !== "all")
            $("records-panel").open = true;
        $("route-view").setAttribute("aria-pressed", String(state.mode === "route"));
        $("all-view").setAttribute("aria-pressed", String(state.mode === "all"));
        $("scope").textContent =
            state.campaign === "all"
                ? "Speed, recorded checks, parent-relative outcomes and integration are separate. Ratios from different campaigns are not directly comparable."
                : `${noteText(campaignInfo(state.campaign)?.notes)} Ratios use this measurement's baseline; imported timing evidence retains its original protocol in the detail panel.`;
        syncControls();
        saveHash();
    }

    function selectFromEvent(event) {
        const highlight = event.target.closest("[data-highlight]");
        if (highlight && byId.has(highlight.dataset.highlight)) {
            const focus = byId.get(highlight.dataset.highlight);
            clearAncestryFocus();
            state.mode = "all";
            state.campaign = focus.campaign;
            state.page = 0;
            setSelection(focus.id);
            return;
        }
        const campaignControl = event.target.closest("[data-campaign]");
        if (campaignControl) {
            clearAncestryFocus();
            state.mode = "all";
            state.campaign = campaignControl.dataset.campaign;
            state.page = 0;
            render();
            return;
        }
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
    $("detail").addEventListener("change", (event) => {
        if (event.target.id === "group-member") setSelection(event.target.value);
    });
    $("candidate-list").addEventListener("click", selectFromEvent);
    $("campaign").addEventListener("change", (event) => {
        clearAncestryFocus();
        state.campaign = event.target.value;
        state.mode = "all";
        state.page = 0;
        render();
    });
    $("outcome").addEventListener("change", (event) => {
        clearAncestryFocus();
        state.outcome = event.target.value;
        state.mode = "all";
        state.page = 0;
        render();
    });
    $("search").addEventListener("input", (event) => {
        clearAncestryFocus();
        state.query = event.target.value.trim().toLowerCase();
        state.mode = "all";
        state.page = 0;
        render();
    });
    for (const [id, key] of [
        [
            "speed",
            "speed",
        ],
        [
            "check-filter",
            "checks",
        ],
    ])
        $(id).addEventListener("change", (event) => {
            clearAncestryFocus();
            state[key] = event.target.value;
            state.page = 0;
            render();
        });
    $("group-replays").addEventListener("change", (event) => {
        state.grouped = event.target.checked;
        state.page = 0;
        render();
    });
    $("page-prev").addEventListener("click", () => {
        state.page = Math.max(0, state.page - 1);
        render();
    });
    $("page-next").addEventListener("click", () => {
        state.page++;
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
        state.speed = "all";
        state.checks = "all";
        state.page = 0;
        state.query = "";
        $("campaign").value = "all";
        $("outcome").value = "all";
        $("search").value = "";
        render();
    });
    $("all-view").addEventListener("click", () => {
        clearAncestryFocus();
        state.mode = "all";
        state.campaign = "all";
        state.page = 0;
        $("records-panel").open = false;
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
            mode: "all",
            campaign: "all",
            query: "",
            outcome: "all",
            speed: "all",
            checks: "all",
            grouped: true,
            page: 0,
            selected: null,
            ancestry: false,
            zoom: 1,
        });
        $("campaign").value = "all";
        $("outcome").value = "all";
        $("search").value = "";
        $("ancestry").checked = false;
        $("records-panel").open = false;
        render();
        $("graph-scroll").scrollTo(0, 0);
    });
    // Theme: follows the system setting until the reader chooses one; the choice is remembered.
    const themeKey = "falcata-lineage-theme";
    const systemDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    const storedTheme = () => {
        try {
            const value = window.localStorage.getItem(themeKey);
            return value === "dark" || value === "light" ? value : null;
        } catch {
            return null;
        }
    };
    function applyTheme(theme) {
        if (theme) document.documentElement.dataset.theme = theme;
        else delete document.documentElement.dataset.theme;
        const dark = (theme || (systemDark?.matches ? "dark" : "light")) === "dark";
        $("theme").textContent = dark ? "Light theme" : "Dark theme";
        $("theme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    }
    applyTheme(storedTheme());
    systemDark?.addEventListener?.("change", () => applyTheme(storedTheme()));
    $("theme").addEventListener("click", () => {
        const current = storedTheme() || (systemDark?.matches ? "dark" : "light");
        const next = current === "dark" ? "light" : "dark";
        try {
            window.localStorage.setItem(themeKey, next);
        } catch {
            // Storage unavailable (private mode or file policy): the choice lasts for this page only.
        }
        applyTheme(next);
    });
    for (const campaign of [
        ...campaignList,
        ...pseudoCampaigns.values(),
    ])
        $("campaign").insertAdjacentHTML(
            "beforeend",
            `<option value="${esc(campaign.id)}">${esc(campaign.label)}${dateRange(campaign) ? ` · ${esc(dateRange(campaign))}` : ""}</option>`,
        );
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
            allGroups.size,
            "entries after source grouping",
        ],
        [
            milestones.length,
            "integration milestones",
        ],
    ];
    $("stats").innerHTML = stats
        .map(([value, text]) => `<div class="stat"><strong>${value}</strong><span>${text}</span></div>`)
        .join("");
    $("limits").innerHTML = [
        ...(data.limitations || []).map((limit) => `<li>${esc(limit)}</li>`),
        ...(data.exclusions || []).map(
            (exclusion) =>
                `<li><strong>Excluded:</strong> ${esc(exclusion.campaign)} — ${esc(exclusion.reason)}${Number.isInteger(exclusion.count) ? ` (${exclusion.count} archive record${exclusion.count === 1 ? "" : "s"})` : ""}.</li>`,
        ),
        ...(data.unknown_edges || []).map(
            (unknown) =>
                `<li><strong>Unresolved reference:</strong> ${esc(unknown.node)} records <code>${esc(unknown.field)}</code> = ${esc(unknown.local_reference)}; ${esc(unknown.reason)}${unknown.choices?.length ? ` (candidates: ${esc(unknown.choices.join(", "))})` : ""}. It is shown as a reference node, not as a measured record.</li>`,
        ),
    ].join("");
    const outcomeValues = [
        ...$("outcome").options,
    ]
        .map((option) => option.value)
        .filter((value) => value !== "all");
    function restoreHash() {
        const hash = new URLSearchParams(location.hash.slice(1));
        state.selected = byId.has(hash.get("candidate")) ? hash.get("candidate") : null;
        state.mode = hash.get("view") === "route" ? "route" : "all";
        state.campaign = campaignInfo(hash.get("campaign"))
            ? hash.get("campaign")
            : state.selected && state.mode === "all" && !hash.has("campaign")
              ? byId.get(state.selected).campaign || "all"
              : "all";
        state.query = (hash.get("query") || "").toLowerCase();
        for (const [key, values] of [
            [
                "speed",
                [
                    "faster",
                    "slower",
                    "unscored",
                ],
            ],
            [
                "checks",
                [
                    "passed",
                    "blocked",
                    "unknown",
                ],
            ],
            [
                "outcome",
                outcomeValues,
            ],
        ])
            state[key] = values.includes(hash.get(key)) ? hash.get(key) : "all";
        state.page = Math.max(0, Number.parseInt(hash.get("page"), 10) || 0);
        state.grouped = hash.get("grouped") !== "false";
        state.ancestry = hash.get("ancestry") === "true";
    }
    window.addEventListener("hashchange", () => {
        if (location.hash === "#records-panel") return;
        restoreHash();
        render();
    });
    restoreHash();
    render();
})();
