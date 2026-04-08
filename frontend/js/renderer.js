/* ═══════════════════════════════════════════════════════════
   Canvas Renderer — Fantasy-themed arena with particle effects
   ═══════════════════════════════════════════════════════════ */

class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.W = canvas.width;
        this.H = canvas.height;
        this.cx = this.W / 2;
        this.cy = this.H / 2;

        // Scale: map world coords → canvas pixels
        this.scale = (this.W / 2 - 30) / 200;   // arena_radius=200, 30px margin

        this.particles = [];
        this.trails1 = [];
        this.trails2 = [];
        this.runeAngle = 0;

        this.bot1Color = '#4fc3f7';
        this.bot2Color = '#e040fb';
        this.goldColor = '#c9a84c';

        // Flash effect
        this.flashIntensity = 0;
        this.flashX = 0;
        this.flashY = 0;

        // Ambient floating particles
        this.ambientParticles = [];
        for (let i = 0; i < 40; i++) {
            this.ambientParticles.push({
                x: Math.random() * this.W,
                y: Math.random() * this.H,
                r: Math.random() * 1.5 + 0.5,
                speed: Math.random() * 0.3 + 0.1,
                phase: Math.random() * Math.PI * 2,
            });
        }

        // Start idle animation
        this._idle = true;
        this._animateIdle();
    }

    // ── coordinate conversion ──────────────────────────────
    wx(worldX) { return this.cx + worldX * this.scale; }
    wy(worldY) { return this.cy + worldY * this.scale; }
    wr(worldR) { return worldR * this.scale; }

    // ── idle animation (when no match is playing) ──────────
    _animateIdle() {
        if (!this._idle) return;
        this.runeAngle += 0.003;
        this._drawBackground();
        this._drawArena(200);
        this._drawAmbient();
        requestAnimationFrame(() => this._animateIdle());
    }

    startMatch() { this._idle = false; }

    stopMatch() {
        this._idle = true;
        this._animateIdle();
    }

    // ── render a single frame ──────────────────────────────
    renderFrame(data) {
        const arenaR = data.arena_radius || 200;
        const bots = data.bots;
        this.runeAngle += 0.005;

        this._drawBackground();
        this._drawArena(arenaR);
        this._drawAmbient();

        // Trails
        this._addTrail(this.trails1, bots[0], this.bot1Color);
        this._addTrail(this.trails2, bots[1], this.bot2Color);
        this._drawTrails(this.trails1);
        this._drawTrails(this.trails2);

        // Collision sparks
        if (data.collision) {
            const mx = (bots[0].x + bots[1].x) / 2;
            const my = (bots[0].y + bots[1].y) / 2;
            this._spawnCollisionParticles(mx, my);
            this.flashIntensity = 1.0;
            this.flashX = this.wx(mx);
            this.flashY = this.wy(my);

            if (window.audio) window.audio.playCollision();

            // Trigger screen shake
            const wrapper = this.canvas.parentElement;
            if (wrapper) {
                wrapper.classList.remove('shake');
                void wrapper.offsetWidth; // trigger reflow
                wrapper.classList.add('shake');
            }
        }
        this._updateAndDrawParticles();

        // Flash overlay
        if (this.flashIntensity > 0) {
            const ctx = this.ctx;
            ctx.save();
            ctx.globalCompositeOperation = 'screen';
            const grad = ctx.createRadialGradient(this.flashX, this.flashY, 0, this.flashX, this.flashY, 200);
            grad.addColorStop(0, `rgba(255, 255, 255, ${this.flashIntensity * 0.6})`);
            grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, this.W, this.H);
            ctx.restore();
            this.flashIntensity -= 0.04;
        }

        // Bots
        this._drawBot(bots[0], this.bot1Color, data.actions ? data.actions[0] : -1);
        this._drawBot(bots[1], this.bot2Color, data.actions ? data.actions[1] : -1);

        // Step counter
        this.ctx.fillStyle = 'rgba(200,190,220,0.4)';
        this.ctx.font = '12px Inter, sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText(`Step ${data.step}`, this.W - 14, this.H - 10);
    }

    // ── background ─────────────────────────────────────────
    _drawBackground() {
        const ctx = this.ctx;
        ctx.fillStyle = '#08080f';
        ctx.fillRect(0, 0, this.W, this.H);

        // Subtle radial ambiance
        const g = ctx.createRadialGradient(this.cx, this.cy, 0, this.cx, this.cy, this.W * 0.6);
        g.addColorStop(0, 'rgba(124,77,255,0.04)');
        g.addColorStop(0.5, 'rgba(100,50,180,0.02)');
        g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, this.W, this.H);
    }

    // ── arena ring ─────────────────────────────────────────
    _drawArena(arenaR) {
        const ctx = this.ctx;
        const r = this.wr(arenaR);

        // Floor
        const floorG = ctx.createRadialGradient(this.cx, this.cy, 0, this.cx, this.cy, r);
        floorG.addColorStop(0, 'rgba(30,20,60,0.5)');
        floorG.addColorStop(0.85, 'rgba(20,14,45,0.6)');
        floorG.addColorStop(1, 'rgba(14,10,30,0.8)');
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, r, 0, Math.PI * 2);
        ctx.fillStyle = floorG;
        ctx.fill();

        // Inner ring glow
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(201,168,76,0.2)';
        ctx.lineWidth = 12;
        ctx.stroke();

        // Edge ring
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = this.goldColor;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Outer glow
        ctx.save();
        ctx.shadowColor = 'rgba(201,168,76,0.35)';
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, r + 1, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(201,168,76,0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        // Rune markers (8 glowing dots around the edge)
        const numRunes = 8;
        for (let i = 0; i < numRunes; i++) {
            const a = this.runeAngle + (i / numRunes) * Math.PI * 2;
            const rx = this.cx + Math.cos(a) * r;
            const ry = this.cy + Math.sin(a) * r;

            ctx.save();
            ctx.shadowColor = this.goldColor;
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(rx, ry, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = this.goldColor;
            ctx.fill();
            ctx.restore();
        }

        // Center marker
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(201,168,76,0.15)';
        ctx.fill();
    }

    // ── bot ────────────────────────────────────────────────
    _drawBot(bot, color, action) {
        const ctx = this.ctx;
        const x = this.wx(bot.x);
        const y = this.wy(bot.y);
        const r = this.wr(bot.radius);

        // Thrust indicator
        const forces = [
            null, [0, -1], [.707, -.707], [1, 0], [.707, .707],
            [0, 1], [-.707, .707], [-1, 0], [-.707, -.707]
        ];
        if (typeof action === 'number' && action > 0 && action <= 8 && forces[action]) {
            const f = forces[action];
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - f[0] * r * 2.5, y - f[1] * r * 2.5);
            ctx.strokeStyle = color + '40';
            ctx.lineWidth = 3;
            ctx.stroke();
        } else if (Array.isArray(action) && (Math.abs(action[0]) > 0.05 || Math.abs(action[1]) > 0.05)) {
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x - action[0] * r * 2.5, y - action[1] * r * 2.5);
            ctx.strokeStyle = color + '40';
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        // Outer glow
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 22;

        // Body
        const bodyG = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
        bodyG.addColorStop(0, '#ffffff');
        bodyG.addColorStop(0.3, color);
        bodyG.addColorStop(1, color + '80');
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = bodyG;
        ctx.fill();
        ctx.restore();

        // Inner ring
        ctx.beginPath();
        ctx.arc(x, y, r * 0.55, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // ── trails ─────────────────────────────────────────────
    _addTrail(arr, bot, color) {
        arr.push({
            x: this.wx(bot.x),
            y: this.wy(bot.y),
            alpha: 0.6,
            r: this.wr(bot.radius) * 0.5,
            color,
        });
        if (arr.length > 18) arr.shift();
    }

    _drawTrails(arr) {
        const ctx = this.ctx;
        arr.forEach((t, i) => {
            t.alpha *= 0.88;
            t.r *= 0.95;
            if (t.alpha < 0.02) return;
            ctx.beginPath();
            ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2);
            ctx.fillStyle = t.color + Math.round(t.alpha * 255).toString(16).padStart(2, '0');
            ctx.fill();
        });
    }

    // ── collision particles ────────────────────────────────
    _spawnCollisionParticles(wx, wy) {
        const x = this.wx(wx);
        const y = this.wy(wy);
        for (let i = 0; i < 18; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 3 + 1.5;
            this.particles.push({
                x, y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 1.0,
                r: Math.random() * 2.5 + 1,
                color: Math.random() > 0.5 ? this.goldColor : '#fff',
            });
        }
    }

    _updateAndDrawParticles() {
        const ctx = this.ctx;
        this.particles = this.particles.filter(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.life -= 0.035;
            p.vx *= 0.97;
            p.vy *= 0.97;
            if (p.life <= 0) return false;
            ctx.save();
            ctx.globalAlpha = p.life;
            ctx.shadowColor = p.color;
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r * p.life, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
            ctx.restore();
            return true;
        });
    }

    // ── ambient floating particles ─────────────────────────
    _drawAmbient() {
        const ctx = this.ctx;
        const t = Date.now() / 1000;
        this.ambientParticles.forEach(p => {
            p.y -= p.speed;
            if (p.y < -5) { p.y = this.H + 5; p.x = Math.random() * this.W; }
            const ox = Math.sin(t + p.phase) * 8;
            ctx.beginPath();
            ctx.arc(p.x + ox, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(201,168,76,${0.12 + Math.sin(t * 2 + p.phase) * 0.06})`;
            ctx.fill();
        });
    }

    // ── clear trails between matches ───────────────────────
    clearTrails() {
        this.trails1.length = 0;
        this.trails2.length = 0;
        this.particles.length = 0;
    }
}

window.Renderer = Renderer;
