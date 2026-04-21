/* ═══════════════════════════════════════════════════════════
   Audio Engine — Procedural Web Audio API sound effects
   ═══════════════════════════════════════════════════════════ */

class AudioEngine {
    constructor() {
        this.ctx = null;
        this.muted = false;
        this.masterGain = null;

        // Don't initialize AudioContext until first user interaction
        const initAudio = () => {
            if (!this.ctx) {
                this.ctx = new (window.AudioContext || window.webkitAudioContext)();
                this.masterGain = this.ctx.createGain();
                this.masterGain.gain.value = 0.3; // Default volume 30%
                this.masterGain.connect(this.ctx.destination);
            }
            document.removeEventListener('click', initAudio);
            document.removeEventListener('keydown', initAudio);
        };

        document.addEventListener('click', initAudio);
        document.addEventListener('keydown', initAudio);
    }

    toggleMute() {
        this.muted = !this.muted;
        if (this.masterGain) {
            this.masterGain.gain.value = this.muted ? 0 : 0.3;
        }
        return this.muted;
    }

    _playTone(freq, type, duration, vol, slideToFreq = null) {
        if (!this.ctx || this.muted) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = type;
        osc.connect(gain);
        gain.connect(this.masterGain);

        const now = this.ctx.currentTime;

        osc.frequency.setValueAtTime(freq, now);
        if (slideToFreq) {
            osc.frequency.exponentialRampToValueAtTime(slideToFreq, now + duration);
        }

        gain.gain.setValueAtTime(vol, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + duration);

        osc.start(now);
        osc.stop(now + duration);

        // Cleanup wrapper to avoid leak
        setTimeout(() => gain.disconnect(), duration * 1000 + 100);
    }

    // Heavy, low thud for collisions
    playCollision() {
        if (!this.ctx || this.muted) return;

        const now = this.ctx.currentTime;

        // Low frequency thud
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(120, now);
        osc.frequency.exponentialRampToValueAtTime(40, now + 0.3);

        // Adding a bit of noise/distortion for impact
        const noise = this.ctx.createBufferSource();
        const bufferSize = this.ctx.sampleRate * 0.1;
        const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
        noise.buffer = buffer;

        const noiseFilter = this.ctx.createBiquadFilter();
        noiseFilter.type = 'lowpass';
        noiseFilter.frequency.value = 800;

        const noiseGain = this.ctx.createGain();
        noiseGain.gain.setValueAtTime(0.4, now);
        noiseGain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);

        noise.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(this.masterGain);

        gain.gain.setValueAtTime(0.6, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
        osc.connect(gain);
        gain.connect(this.masterGain);

        osc.start(now);
        noise.start(now);
        osc.stop(now + 0.3);

        setTimeout(() => {
            gain.disconnect();
            noiseFilter.disconnect();
            noiseGain.disconnect();
        }, 400);
    }

    playWin() {
        this._playTone(440, 'triangle', 0.2, 0.4);            // A4
        setTimeout(() => this._playTone(554, 'triangle', 0.2, 0.4), 150); // C#5
        setTimeout(() => this._playTone(659, 'triangle', 0.6, 0.5), 300); // E5
    }

    playLoss() {
        this._playTone(300, 'sawtooth', 0.3, 0.3, 150);
        setTimeout(() => this._playTone(250, 'sawtooth', 0.5, 0.3, 100), 300);
    }
}

window.audio = new AudioEngine();
