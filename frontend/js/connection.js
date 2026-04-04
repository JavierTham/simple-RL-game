/* ═══════════════════════════════════════════════════════════
   WebSocket connection manager
   ═══════════════════════════════════════════════════════════ */

class Connection {
    constructor() {
        this.ws = null;
        this.handlers = {};
        this.reconnectDelay = 1000;
        this._connect();
    }

    _connect() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        this.ws = new WebSocket(`${proto}://${location.host}/ws`);

        this.ws.onopen = () => {
            console.log('[WS] Connected');
            this.reconnectDelay = 1000;
            this._dispatch('open', null);
        };

        this.ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                this._dispatch(data.type, data);
            } catch (err) {
                console.error('[WS] Parse error', err);
            }
        };

        this.ws.onclose = () => {
            console.log('[WS] Disconnected, reconnecting...');
            this._dispatch('close', null);
            setTimeout(() => this._connect(), this.reconnectDelay);
            this.reconnectDelay = Math.min(this.reconnectDelay * 2, 8000);
        };

        this.ws.onerror = (err) => {
            console.error('[WS] Error', err);
        };
    }

    send(type, payload = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, ...payload }));
        }
    }

    on(type, handler) {
        if (!this.handlers[type]) this.handlers[type] = [];
        this.handlers[type].push(handler);
    }

    off(type, handler) {
        if (this.handlers[type]) {
            this.handlers[type] = this.handlers[type].filter(h => h !== handler);
        }
    }

    _dispatch(type, data) {
        (this.handlers[type] || []).forEach(h => h(data));
    }
}

// Singleton
window.conn = new Connection();
