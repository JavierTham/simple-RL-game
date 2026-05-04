/* ═══════════════════════════════════════════════════════════
   App — Main UI logic, wiring, and state management
   ═══════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── state ──────────────────────────────────────────────
    let trainedWeights = null;  // weights of last trained bot
    let pvpBot1Weights = null;
    let pvpBot2Weights = null;
    let tourneyBotsWeights = [null, null, null, null];
    let tourneyBotNames = ['', '', '', ''];
    let tourneyPhase = 0; // 0: inactive, 1: SF1, 2: SF2, 3: Final
    let tourneyScores = [0, 0];
    let tourneyRoundsPlayed = 0;
    let tourneyRoundsTotal = 3;
    let tourneyWinners = ['', ''];
    let tourneyWinnerWeights = [null, null];
    let isTraining = false;
    let isPlaying = false;      // match is being streamed
    let pvpScores = [0, 0];
    let pvpRoundsTotal = 3;
    let pvpRoundsPlayed = 0;

    // ── DOM refs ───────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const canvas = $('#arena-canvas');
    const renderer = new Renderer(canvas);

    // Tabs
    const tabs = $$('.tab');
    const panels = $$('.tab-content');

    // Train
    const btnTrain = $('#btn-train');
    const btnStopTrain = $('#btn-stop-train');
    const progressContainer = $('#progress-container');
    const progressFill = $('#progress-fill');
    const progressText = $('#progress-text');
    const statsGrid = $('#stats-grid');
    const chartContainer = $('#chart-container');
    const postTrainActions = $('#post-train-actions');

    // Chart Component
    const liveChart = window.LiveChart ? new window.LiveChart('training-chart') : null;

    // Test
    const btnTest = $('#btn-test');
    const testResult = $('#test-result');
    const testResultText = $('#test-result-text');

    // PvP
    const btnPvp = $('#btn-pvp');
    const pvpScoreboard = $('#pvp-scoreboard');
    const pvpResult = $('#pvp-result');
    const pvpResultText = $('#pvp-result-text');

    // Tourney
    const btnTourney = $('#btn-start-tourney');
    const tourneyBracket = $('#tourney-bracket');
    const tourneyStatusBox = $('#tourney-status-box');
    const tourneyStatusText = $('#tourney-status-text');

    // Save modal
    const saveModal = $('#save-modal');
    const saveNameInput = $('#save-name-input');
    const btnSave = $('#btn-save');
    const btnSaveConfirm = $('#btn-save-confirm');
    const btnSaveCancel = $('#btn-save-cancel');
    const btnDownload = $('#btn-download');

    // Overlay
    const matchOverlay = $('#match-overlay');
    const overlayText = $('#overlay-text');
    const statusCenter = $('#status-center');

    // Audio
    const btnAudioToggle = $('#btn-audio-toggle');
    if (btnAudioToggle) {
        btnAudioToggle.addEventListener('click', () => {
            if (window.audio) {
                const isMuted = window.audio.toggleMute();
                btnAudioToggle.classList.toggle('muted', isMuted);
                btnAudioToggle.textContent = isMuted ? '🔇' : '🔊';
            }
        });
    }

    // ── helpers ─────────────────────────────────────────────
    function toast(msg, type = '') {
        const el = document.createElement('div');
        el.className = 'toast ' + type;
        el.textContent = msg;
        $('#toast-container').appendChild(el);
        setTimeout(() => el.remove(), 3500);
    }

    function showOverlay(text, duration = 2000) {
        overlayText.textContent = text;
        matchOverlay.classList.add('visible');
        if (duration > 0) {
            setTimeout(() => matchOverlay.classList.remove('visible'), duration);
        }
    }

    function getRewardWeights() {
        const keys = ['win_bonus', 'charge_reward', 'hit_reward', 'opp_edge', 'edge_penalty'];
        const w = {};
        keys.forEach(k => {
            w[k] = parseFloat($(`#slider-${k}`).value);
        });
        return w;
    }

    // ── tab switching ──────────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            $(`#panel-${tab.dataset.tab}`).classList.add('active');

            // Refresh bot list when switching to PvP
            if (tab.dataset.tab === 'pvp') {
                conn.send('list_bots');
            }
        });
    });

    // ── slider live values ─────────────────────────────────
    $$('input[type="range"]').forEach(slider => {
        const id = slider.id.replace('slider-', '');
        slider.addEventListener('input', () => {
            // $(`#val-${id}`).textContent = parseFloat(slider.value).toFixed(1);
            $(`#val-${id}`).textContent =
                id === "input_episodes"
                    ? parseInt(slider.value)
                    : parseFloat(slider.value).toFixed(1);
            // Deactivate preset buttons when user manually adjusts
            $$('.preset-btn').forEach(b => b.classList.remove('active'));
        });
    });

    const PRESETS = {
        // ── Intended behavior ──
        balanced: { win_bonus: 1.0, charge_reward: 1.0, hit_reward: 1.0, opp_edge: 1.0, edge_penalty: 1.0 },
        aggressive: { win_bonus: 1.5, charge_reward: 0.5, hit_reward: 2.0, opp_edge: 1.5, edge_penalty: 0.3 },
        // ── Hackable experiments (reward hacking demos) ──
        camper: { win_bonus: 0.2, charge_reward: 0.0, hit_reward: 0.0, opp_edge: 0.0, edge_penalty: 2.0 },
        bumper: { win_bonus: 0.1, charge_reward: 2.0, hit_reward: 2.0, opp_edge: 0.0, edge_penalty: 0.0 },
        stalker: { win_bonus: 0.0, charge_reward: 0.5, hit_reward: 0.0, opp_edge: 2.0, edge_penalty: 0.5 },
    };

    function applyPreset(name) {
        const p = PRESETS[name];
        if (!p) return;
        Object.entries(p).forEach(([key, val]) => {
            const slider = $(`#slider-${key}`);
            if (slider) {
                slider.value = val;
                $(`#val-${key}`).textContent = val.toFixed(1);
            }
        });
        $$('.preset-btn').forEach(b => b.classList.remove('active'));
        const activeBtn = $(`.preset-btn[data-preset="${name}"]`);
        if (activeBtn) activeBtn.classList.add('active');
    }

    $$('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });

    // ── training ───────────────────────────────────────────
    btnTrain.addEventListener('click', () => {
        if (isTraining) return;
        isTraining = true;

        const config = {
            num_episodes: parseInt($('#slider-input_episodes').value),
            learning_rate: parseFloat($('#input-lr').value),
            reward_weights: getRewardWeights(),
        };

        conn.send('start_training', { config });

        btnTrain.classList.add('hidden');
        btnStopTrain.classList.remove('hidden');
        progressContainer.classList.remove('hidden');
        statsGrid.classList.remove('hidden');
        if (chartContainer) chartContainer.classList.remove('hidden');
        postTrainActions.classList.add('hidden');
        statusCenter.textContent = 'Training...';
        toast('Training started ⚗️');

        if (liveChart) liveChart.reset(config.num_episodes);
    });

    btnStopTrain.addEventListener('click', () => {
        conn.send('stop_training');
        toast('Stopping training...');
    });

    conn.on('training_progress', (data) => {
        const pct = ((data.episode / data.total_episodes) * 100).toFixed(1);
        progressFill.style.width = pct + '%';
        progressText.textContent = `${data.episode} / ${data.total_episodes} (${pct}%)`;
        $('#stat-winrate').textContent = (data.win_rate * 100).toFixed(1) + '%';
        $('#stat-episode').textContent = data.episode;
        $('#stat-wins').textContent = (data.lose_rate * 100).toFixed(1) + '%';
        $('#stat-reward').textContent = data.avg_reward.toFixed(2);

        if (liveChart) {
            liveChart.push(data.win_rate, data.lose_rate);
        }
    });

    conn.on('training_complete', (data) => {
        isTraining = false;
        trainedWeights = data.weights;
        btnTrain.classList.remove('hidden');
        btnStopTrain.classList.add('hidden');
        postTrainActions.classList.remove('hidden');
        statusCenter.textContent = 'Training Complete ✓';
        toast('Training complete! 🎉', 'success');
    });

    // ── save bot ───────────────────────────────────────────
    btnSave.addEventListener('click', () => {
        if (!trainedWeights) { toast('No trained bot to save', 'error'); return; }
        saveModal.classList.remove('hidden');
        saveNameInput.value = '';
        saveNameInput.focus();
    });

    btnSaveCancel.addEventListener('click', () => saveModal.classList.add('hidden'));

    btnSaveConfirm.addEventListener('click', () => {
        const name = saveNameInput.value.trim() || 'unnamed_bot';
        conn.send('save_bot', { name, weights: trainedWeights });
        saveModal.classList.add('hidden');
    });

    btnDownload.addEventListener('click', () => {
        if (!trainedWeights) return;
        const name = saveNameInput.value.trim() || 'bot';
        const blob = new Blob([JSON.stringify({ name, weights: trainedWeights })], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = name + '.json';
        a.click();
        URL.revokeObjectURL(a.href);
        saveModal.classList.add('hidden');
        toast('Bot downloaded 📥', 'success');
    });

    conn.on('bot_saved', (data) => {
        toast(`Bot "${data.name}" saved! 💾`, 'success');
    });

    // ── test match ─────────────────────────────────────────
    btnTest.addEventListener('click', () => {
        if (!trainedWeights) { toast('Train a bot first!', 'error'); return; }
        if (isPlaying) return;
        isPlaying = true;
        btnTest.disabled = true;
        testResult.classList.add('hidden');
        renderer.clearTrails();
        renderer.startMatch();
        statusCenter.textContent = 'Test Match';
        showOverlay('⚔️ Test Match!', 1500);

        const speed = parseFloat($('#test-speed').value);
        conn.send('test_match', { reward_weights: getRewardWeights(), speed });
    });

    // ── PvP & Tourney ────────────────────────────────────────────────
    conn.on('bot_list', (data) => {
        ['pvp-bot1-select', 'pvp-bot2-select', 'tourney-bot1', 'tourney-bot2', 'tourney-bot3', 'tourney-bot4'].forEach(selId => {
            const sel = $(`#${selId}`);
            if (!sel) return;
            const current = sel.value;
            sel.innerHTML = '<option value="">-- select --</option>';
            data.bots.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name; opt.textContent = name;
                sel.appendChild(opt);
            });
            if (current) sel.value = current;
        });

        // Render bot manager list
        const listEl = $('#bot-manager-list');
        if (listEl) {
            listEl.innerHTML = '';
            data.bots.forEach(name => {
                const row = document.createElement('div');
                row.className = 'bot-manager-row';
                row.innerHTML = `<span class="bot-manager-name">${name}</span>
                    <button class="btn-delete-bot" data-bot="${name}" title="Delete ${name}">✕</button>`;
                listEl.appendChild(row);
            });
            listEl.querySelectorAll('.btn-delete-bot').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (confirm(`Delete bot "${btn.dataset.bot}"?`)) {
                        conn.send('delete_bot', { name: btn.dataset.bot });
                    }
                });
            });
        }
    });

    conn.on('bot_deleted', (data) => {
        toast(`Bot "${data.name}" deleted 🗑️`, 'success');
    });

    // Load from server dropdown
    $('#pvp-bot1-select').addEventListener('change', function () {
        if (this.value) conn.send('load_bot', { name: this.value, slot: 1 });
    });
    $('#pvp-bot2-select').addEventListener('change', function () {
        if (this.value) conn.send('load_bot', { name: this.value, slot: 2 });
    });

    $('#tourney-bot1')?.addEventListener('change', function () { if (this.value) conn.send('load_bot', { name: this.value, slot: 11 }); });
    $('#tourney-bot2')?.addEventListener('change', function () { if (this.value) conn.send('load_bot', { name: this.value, slot: 12 }); });
    $('#tourney-bot3')?.addEventListener('change', function () { if (this.value) conn.send('load_bot', { name: this.value, slot: 13 }); });
    $('#tourney-bot4')?.addEventListener('change', function () { if (this.value) conn.send('load_bot', { name: this.value, slot: 14 }); });

    // Upload from file
    $('#pvp-bot1-upload').addEventListener('change', (e) => loadBotFromFile(e, 1));
    $('#pvp-bot2-upload').addEventListener('change', (e) => loadBotFromFile(e, 2));

    function loadBotFromFile(e, slot) {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const data = JSON.parse(ev.target.result);
                if (slot === 1) {
                    pvpBot1Weights = data.weights;
                    $('#pvp-bot1-status').textContent = `✓ "${data.name || file.name}" loaded`;
                } else {
                    pvpBot2Weights = data.weights;
                    $('#pvp-bot2-status').textContent = `✓ "${data.name || file.name}" loaded`;
                }
                toast(`Bot loaded from file 📂`, 'success');
            } catch {
                toast('Invalid JSON file', 'error');
            }
        };
        reader.readAsText(file);
    }

    conn.on('bot_loaded', (data) => {
        if (data.slot === 1) {
            pvpBot1Weights = data.data.weights;
            $('#pvp-bot1-status').textContent = `✓ "${data.data.name}" loaded`;
        } else if (data.slot === 2) {
            pvpBot2Weights = data.data.weights;
            $('#pvp-bot2-status').textContent = `✓ "${data.data.name}" loaded`;
        } else if (data.slot >= 11 && data.slot <= 14) {
            let idx = data.slot - 11;
            tourneyBotsWeights[idx] = data.data.weights;
            tourneyBotNames[idx] = data.data.name;
        }
        toast(`Bot "${data.data.name}" loaded`, 'success');
    });

    btnPvp.addEventListener('click', () => {
        if (!pvpBot1Weights || !pvpBot2Weights) {
            toast('Load both bots first!', 'error'); return;
        }
        if (isPlaying) return;

        pvpRoundsTotal = parseInt($('#pvp-rounds').value);
        pvpScores = [0, 0];
        pvpRoundsPlayed = 0;
        updatePvpScoreboard();
        pvpScoreboard.classList.remove('hidden');
        pvpResult.classList.add('hidden');
        btnPvp.disabled = true;

        startPvpRound();
    });

    function startPvpRound() {
        isPlaying = true;
        renderer.clearTrails();
        renderer.startMatch();
        const roundNum = pvpRoundsPlayed + 1;
        statusCenter.textContent = `PvP Round ${roundNum}`;
        showOverlay(`Round ${roundNum}`, 1200);

        const speed = parseFloat($('#pvp-speed').value);
        conn.send('pvp_match', { bot1_weights: pvpBot1Weights, bot2_weights: pvpBot2Weights, speed });
    }

    function updatePvpScoreboard() {
        $('#pvp-score1').textContent = pvpScores[0];
        $('#pvp-score2').textContent = pvpScores[1];
    }

    // ── match frame & result handlers ──────────────────────
    conn.on('match_frame', (data) => {
        renderer.renderFrame(data);
    });

    conn.on('match_result', (data) => {
        isPlaying = false;
        renderer.stopMatch();

        const winnerText = data.winner === 0 ? 'Bot 1 (Blue) Wins!'
            : data.winner === 1 ? 'Bot 2 (Red) Wins!'
                : 'Draw!';

        // Determine which mode we're in
        const activeTab = document.querySelector('.tab.active').dataset.tab;

        if (activeTab === 'test') {
            const resultStr = data.winner === 0 ? '🏆 Your Bot Wins!' : data.winner === 1 ? '💀 Default Bot Wins' : '🤝 Draw';
            if (data.winner === 0 && window.audio) window.audio.playWin();
            else if (data.winner === 1 && window.audio) window.audio.playLoss();

            testResultText.textContent = resultStr + ` (${data.steps} steps)`;
            testResult.classList.remove('hidden');
            btnTest.disabled = false;
            statusCenter.textContent = resultStr;
            showOverlay(resultStr, 2500);

        } else if (activeTab === 'pvp') {
            pvpRoundsPlayed++;
            if (data.winner === 0) pvpScores[0]++;
            else if (data.winner === 1) pvpScores[1]++;
            updatePvpScoreboard();
            showOverlay(winnerText, 1500);

            const winsNeeded = Math.ceil(pvpRoundsTotal / 2);
            if (pvpScores[0] >= winsNeeded || pvpScores[1] >= winsNeeded || pvpRoundsPlayed >= pvpRoundsTotal) {
                // Series over
                const champion = pvpScores[0] > pvpScores[1] ? 'Bot 1 (Blue)' : pvpScores[1] > pvpScores[0] ? 'Bot 2 (Red)' : 'Nobody';
                if (pvpScores[0] !== pvpScores[1] && window.audio) window.audio.playWin();

                pvpResultText.textContent = `🏆 ${champion} wins the series ${pvpScores[0]}-${pvpScores[1]}!`;
                pvpResult.classList.remove('hidden');
                btnPvp.disabled = false;
                statusCenter.textContent = `${champion} is Champion!`;
            } else {
                // Next round after delay
                setTimeout(() => startPvpRound(), 2000);
            }
        } else if (activeTab === 'tourney') {
            tourneyRoundsPlayed++;
            if (data.winner === 0) tourneyScores[0]++;
            else if (data.winner === 1) tourneyScores[1]++;

            showOverlay(winnerText, 1500);

            const winsNeeded = Math.ceil(tourneyRoundsTotal / 2);
            if (tourneyScores[0] >= winsNeeded || tourneyScores[1] >= winsNeeded || tourneyRoundsPlayed >= tourneyRoundsTotal) {
                // Determine series winner
                const wIdx = tourneyScores[0] > tourneyScores[1] ? 0 : 1;
                let wName = ''; let wW = null;
                if (tourneyPhase === 1) {
                    wName = tourneyBotNames[wIdx]; wW = tourneyBotsWeights[wIdx];
                    tourneyWinners[0] = wName; tourneyWinnerWeights[0] = wW;
                    $('#sf1-p1').classList.toggle('winner', wIdx === 0); $('#sf1-p1').classList.toggle('loser', wIdx !== 0);
                    $('#sf1-p2').classList.toggle('winner', wIdx === 1); $('#sf1-p2').classList.toggle('loser', wIdx !== 1);
                    $('#fin-p1').textContent = wName;
                } else if (tourneyPhase === 2) {
                    wName = tourneyBotNames[2 + wIdx]; wW = tourneyBotsWeights[2 + wIdx];
                    tourneyWinners[1] = wName; tourneyWinnerWeights[1] = wW;
                    $('#sf2-p1').classList.toggle('winner', wIdx === 0); $('#sf2-p1').classList.toggle('loser', wIdx !== 0);
                    $('#sf2-p2').classList.toggle('winner', wIdx === 1); $('#sf2-p2').classList.toggle('loser', wIdx !== 1);
                    $('#fin-p2').textContent = wName;
                } else if (tourneyPhase === 3) {
                    wName = tourneyWinners[wIdx];
                    $('#fin-p1').classList.toggle('winner', wIdx === 0); $('#fin-p1').classList.toggle('loser', wIdx !== 0);
                    $('#fin-p2').classList.toggle('winner', wIdx === 1); $('#fin-p2').classList.toggle('loser', wIdx !== 1);
                    $('#tourney-champ').textContent = `🏆 ${wName} 🏆`;
                    tourneyStatusText.textContent = `👑 THE GRAND CHAMPION IS ${wName} 👑`;
                    tourneyStatusBox.classList.remove('hidden');
                    btnTourney.disabled = false;
                    statusCenter.textContent = `${wName} takes the Crown!`;
                    if (window.audio) window.audio.playWin();
                    tourneyPhase = 0;
                    return;
                }

                // Next phase
                tourneyPhase++;
                setTimeout(() => startTourneySeries(), 2500);
            } else {
                // Next match in same series
                setTimeout(() => runTourneyMatch(), 2000);
            }
        }
    });

    // ── Tourney Handlers ─────────────────────────────────────
    btnTourney?.addEventListener('click', () => {
        if (tourneyBotsWeights.includes(null)) {
            toast('Load 4 bots to start!', 'error'); return;
        }
        if (isPlaying) return;

        btnTourney.disabled = true;
        tourneyBracket.classList.remove('hidden');
        tourneyStatusBox.classList.add('hidden');

        // reset UI
        ['sf1-p1', 'sf1-p2', 'sf2-p1', 'sf2-p2', 'fin-p1', 'fin-p2'].forEach(id => {
            $(`#${id}`).classList.remove('winner', 'loser');
        });
        $('#sf1-p1').textContent = tourneyBotNames[0];
        $('#sf1-p2').textContent = tourneyBotNames[1];
        $('#sf2-p1').textContent = tourneyBotNames[2];
        $('#sf2-p2').textContent = tourneyBotNames[3];
        $('#fin-p1').textContent = 'TBD';
        $('#fin-p2').textContent = 'TBD';
        $('#tourney-champ').textContent = '🏆 ? 🏆';

        tourneyRoundsTotal = parseInt($('#tourney-rounds').value);
        tourneyPhase = 1;
        startTourneySeries();
    });

    function startTourneySeries() {
        tourneyScores = [0, 0];
        tourneyRoundsPlayed = 0;

        $$('.bracket-match').forEach(m => m.classList.remove('active'));
        if (tourneyPhase === 1) {
            $('#tm-sf1').classList.add('active');
            statusCenter.textContent = 'Semifinal 1';
            showOverlay('Semifinal 1', 1500);
        } else if (tourneyPhase === 2) {
            $('#tm-sf2').classList.add('active');
            statusCenter.textContent = 'Semifinal 2';
            showOverlay('Semifinal 2', 1500);
        } else if (tourneyPhase === 3) {
            $('#tm-fin').classList.add('active');
            statusCenter.textContent = 'Grand Final';
            showOverlay('Grand Final', 2000);
        }

        setTimeout(() => runTourneyMatch(), 2000);
    }

    function runTourneyMatch() {
        isPlaying = true;
        renderer.clearTrails();
        renderer.startMatch();
        const speed = parseFloat($('#tourney-speed').value);

        let bw1, bw2;
        if (tourneyPhase === 1) {
            bw1 = tourneyBotsWeights[0]; bw2 = tourneyBotsWeights[1];
        } else if (tourneyPhase === 2) {
            bw1 = tourneyBotsWeights[2]; bw2 = tourneyBotsWeights[3];
        } else {
            bw1 = tourneyWinnerWeights[0]; bw2 = tourneyWinnerWeights[1];
        }

        conn.send('pvp_match', { bot1_weights: bw1, bot2_weights: bw2, speed });
    }

    // ── error handler ──────────────────────────────────────
    conn.on('error', (data) => {
        toast(data.msg || 'Something went wrong', 'error');
        isPlaying = false;
        btnTest.disabled = false;
        btnPvp.disabled = false;
    });

    // ── connection status ──────────────────────────────────
    conn.on('open', () => {
        statusCenter.textContent = 'Connected ✓';
        conn.send('list_bots');
    });
    conn.on('close', () => {
        statusCenter.textContent = 'Disconnected...';
    });

})();
