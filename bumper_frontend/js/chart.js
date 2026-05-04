/* ═══════════════════════════════════════════════════════════
   Chart Renderer — Live visualization for training metrics
   ═══════════════════════════════════════════════════════════ */

class LiveChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.W = this.canvas.width;
        this.H = this.canvas.height;

        this.dataWinRate = [];
        this.dataLossRate = [];
        this.maxLen = 0;

        this.padL = 40;
        this.padR = 10;
        this.padT = 10;
        this.padB = 20;

        this.plotW = this.W - this.padL - this.padR;
        this.plotH = this.H - this.padT - this.padB;

        this.clear();
    }

    reset(maxEpisodes) {
        this.maxLen = maxEpisodes;
        this.dataWinRate = [];
        this.dataLossRate = [];
        this.render();
    }

    push(winRate, lossRate) {
        this.dataWinRate.push(winRate);
        this.dataLossRate.push(lossRate);
        this.render();
    }

    clear() {
        this.ctx.clearRect(0, 0, this.W, this.H);
        this._drawGrid();
    }

    _drawGrid() {
        const ctx = this.ctx;
        ctx.strokeStyle = 'rgba(201,168,76,0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();

        // Y grid lines
        for (let i = 0; i <= 4; i++) {
            const y = this.padT + (i / 4) * this.plotH;
            ctx.moveTo(this.padL, y);
            ctx.lineTo(this.W - this.padR, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#9890a8';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        // Labels for Win/Loss Rate [0, 1] on the left
        ctx.fillText('100%', this.padL - 5, this.padT);
        ctx.fillText('50%', this.padL - 5, this.padT + this.plotH / 2);
        ctx.fillText('0%', this.padL - 5, this.padT + this.plotH);
    }

    render() {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.W, this.H);
        this._drawGrid();

        if (this.dataWinRate.length === 0) return;

        const len = this.dataWinRate.length;
        const total = Math.max(this.maxLen, len);

        const dx = total > 1 ? this.plotW / (total - 1) : 0;

        // Draw Reward (scaled arbitrarily roughly -1 to +2)
        // this.ctx.beginPath();
        // this.ctx.strokeStyle = 'rgba(224,64,251,0.6)';
        // this.ctx.lineWidth = 1.5;
        // for (let i = 0; i < len; i++) {
        //     const x = this.padL + i * dx;
        //     // Map reward from [-1, 2] to [plotH, 0]
        //     let r = this.dataLossRate[i];
        //     r = Math.max(-1, Math.min(2, r));
        //     const y = this.padT + this.plotH - ((r + 1) / 3) * this.plotH;
        //     if (i === 0) this.ctx.moveTo(x, y);
        //     else this.ctx.lineTo(x, y);
        // }
        // this.ctx.stroke();

        // Draw Win Rate (0 to 1)
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#66bb6a'; // green
        this.ctx.lineWidth = 2;
        for (let i = 0; i < len; i++) {
            const x = this.padL + i * dx;
            const w = this.dataWinRate[i];
            const y = this.padT + this.plotH - w * this.plotH;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();

        // Draw Loss Rate (0 to 1)
        this.ctx.beginPath();
        this.ctx.strokeStyle = 'rgba(224,64,251,0.6)'; // purple
        this.ctx.lineWidth = 2;
        for (let i = 0; i < len; i++) {
            const x = this.padL + i * dx;
            const w = this.dataLossRate[i];
            const y = this.padT + this.plotH - w * this.plotH;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();
    }
}

window.LiveChart = LiveChart;
