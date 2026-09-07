/**
 * Voice Agent Web Dashboard
 *
 * Connects to the WebSocket voice gateway, captures microphone audio,
 * streams it to the server, and plays back TTS responses.
 */
class VoiceAgent {
    constructor() {
        this.ws = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.recorder = null;
        this.isRecording = false;
        this.sessionId = null;
        this.language = null;
        this.turnCount = 0;
        this.animFrame = null;

        this.initElements();
        this.initAudio();
        this.initWebSocket();
        this.startLevelMeter();
    }

    // -------------------------------------------------------------------------
    // DOM element bindings
    // -------------------------------------------------------------------------
    initElements() {
        this.micBtn = document.getElementById('mic-btn');
        this.statusEl = document.getElementById('status');
        this.stateEl = document.getElementById('state');
        this.languageEl = document.getElementById('language');
        this.sessionEl = document.getElementById('session');
        this.messagesEl = document.getElementById('messages');
        this.levelMeter = document.getElementById('level-meter');
        this.interruptBtn = document.getElementById('interrupt-btn');

        this.micBtn.addEventListener('mousedown', () => this.startRecording());
        this.micBtn.addEventListener('mouseup', () => this.stopRecording());
        this.micBtn.addEventListener('mouseleave', () => {
            if (this.isRecording) this.stopRecording();
        });
        // Touch support
        this.micBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        this.micBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopRecording();
        });

        this.interruptBtn.addEventListener('click', () => this.sendInterrupt());
    }

    // -------------------------------------------------------------------------
    // Audio setup — microphone + analyser for visualizer
    // -------------------------------------------------------------------------
    async initAudio() {
        try {
            this.audioContext = new AudioContext({ sampleRate: 16000 });
        } catch (e) {
            console.warn('AudioContext not available:', e);
        }
    }

    async initMicrophone() {
        if (this.mediaStream) return;
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
        // Create analyser for visualizer
        if (this.audioContext) {
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
        }
    }

    // -------------------------------------------------------------------------
    // WebSocket
    // -------------------------------------------------------------------------
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = window.location.port || '8765';
        const url = `${protocol}//${host}:${port}`;

        this.ws = new WebSocket(url);
        this.setStatus('Connecting...');

        this.ws.onopen = () => {
            this.setStatus('Connected');
            this.micBtn.disabled = false;
        };

        this.ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                const msg = JSON.parse(event.data);
                this.handleMessage(msg);
            } else {
                // Binary audio response
                this.playAudio(event.data);
            }
        };

        this.ws.onclose = () => {
            this.setStatus('Disconnected');
            this.micBtn.disabled = true;
            this.stopRecording();
            // Auto-reconnect after 3s
            this.setStatus('Reconnecting...');
            setTimeout(() => this.initWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            this.setStatus('Error', 'error');
        };
    }

    // -------------------------------------------------------------------------
    // Message handler
    // -------------------------------------------------------------------------
    handleMessage(msg) {
        switch (msg.type) {
            case 'status':
                this.sessionId = msg.session_id;
                this.sessionEl.textContent = (msg.session_id || '-').slice(0, 8);
                this.setState(msg.state);
                break;

            case 'speech_start':
                this.setState('listening');
                this.addSystemMessage('Speech detected...');
                break;

            case 'speech_end':
                this.setState('transcribing');
                break;

            case 'turn_started':
                this.setState('thinking');
                this.addSystemMessage('Processing response...');
                break;

            case 'turn_complete':
                this.turnCount++;
                document.getElementById('turns').textContent = this.turnCount;
                this.language = msg.language || this.language;
                if (msg.language) this.languageEl.textContent = msg.language;
                if (msg.duration) {
                    this.addSystemMessage(`Response: ${msg.duration.toFixed(1)}s audio`);
                }
                this.setState('idle');
                this.micBtn.disabled = false;
                break;

            case 'error':
                this.addSystemMessage(`Error: ${msg.message}`);
                this.setState('error');
                this.micBtn.disabled = false;
                break;

            case 'pong':
                // Heartbeat response
                break;
        }
    }

    // -------------------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------------------
    async startRecording() {
        if (this.isRecording || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        try {
            await this.initMicrophone();

            // Use ScriptProcessor to capture raw PCM
            const scriptNode = this.audioContext.createScriptProcessor(4096, 1, 1);
            this.audioBuffer = [];

            scriptNode.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                // Convert Float32 [-1, 1] to Int16 PCM
                const pcm = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcm[i] = s < 0 ? s * 32768 : s * 32767;
                }
                this.audioBuffer.push(new Uint8Array(pcm.buffer));
            };

            this.scriptNode = scriptNode;
            this.audioBuffer = [];
            this.isRecording = true;
            this.micBtn.classList.add('recording');
            this.interruptBtn.disabled = false;
            this.setState('listening');
            this.setStatus('Recording', 'recording');

        } catch (err) {
            console.error('Failed to start recording:', err);
            this.setStatus(`Mic error: ${err.message}`, 'error');
        }
    }

    stopRecording() {
        if (!this.isRecording) return;

        this.isRecording = false;
        this.micBtn.classList.remove('recording');
        this.setStatus('Processing...', 'processing');

        // Disconnect script processor
        if (this.scriptNode) {
            this.scriptNode.disconnect();
            this.scriptNode = null;
        }

        if (this.audioBuffer && this.audioBuffer.length > 0 && this.ws) {
            // Concatenate all PCM chunks
            const total = this.audioBuffer.reduce((a, b) => a + b.length, 0);
            const combined = new Uint8Array(total);
            let offset = 0;
            for (const chunk of this.audioBuffer) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            this.ws.send(combined);
            this.audioBuffer = null;
        }
    }

    sendInterrupt() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'interrupt' }));
            this.stopRecording();
            this.setState('interrupted');
            this.addSystemMessage('Interrupted by user');
        }
    }

    // -------------------------------------------------------------------------
    // Audio playback
    // -------------------------------------------------------------------------
    async playAudio(audioData) {
        try {
            // Decode and play
            const audioBuffer = await this.audioContext.decodeAudioData(
                audioData.slice(0)
            );
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
        } catch (e) {
            console.warn('Could not decode audio:', e);
        }
    }

    // -------------------------------------------------------------------------
    // Level meter visualizer (canvas)
    // -------------------------------------------------------------------------
    startLevelMeter() {
        const canvas = this.levelMeter;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Handle high-DPI displays
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        const draw = () => {
            ctx.clearRect(0, 0, width, height);

            // Background bar
            ctx.fillStyle = '#334155';
            ctx.fillRect(0, 0, width, height);

            let level = 0;
            if (this.analyser && this.isRecording) {
                const data = new Uint8Array(this.analyser.frequencyBinCount);
                this.analyser.getByteFrequencyData(data);
                const avg = data.reduce((a, b) => a + b, 0) / data.length;
                level = avg / 255;
            }

            // Level bar (green → yellow → red)
            const barWidth = level * width;
            const hue = 120 - level * 120; // green → yellow → red
            ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
            ctx.fillRect(0, 0, barWidth, height);

            // Threshold line at 50%
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(width * 0.5, 0);
            ctx.lineTo(width * 0.5, height);
            ctx.stroke();

            // Level text
            ctx.fillStyle = '#fff';
            ctx.font = '11px ui-monospace';
            ctx.textAlign = 'right';
            ctx.fillText(`${Math.round(level * 100)}%`, width - 8, height - 8);

            this.animFrame = requestAnimationFrame(draw);
        };

        draw();
    }

    // -------------------------------------------------------------------------
    // UI helpers
    // -------------------------------------------------------------------------
    setStatus(text, cls = '') {
        this.statusEl.textContent = text;
        this.statusEl.className = 'value ' + cls;
    }

    setState(state) {
        this.stateEl.textContent = state;
        this.stateEl.className = 'value ' + state;
    }

    addMessage(role, text, meta = '') {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const content = document.createElement('div');
        content.textContent = text;
        div.appendChild(content);

        if (meta) {
            const metaEl = document.createElement('div');
            metaEl.className = 'message-meta';
            metaEl.textContent = meta;
            div.appendChild(metaEl);
        }

        this.messagesEl.appendChild(div);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    addSystemMessage(text) {
        this.addMessage('system', text);
    }
}

// Initialise when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAgent = new VoiceAgent();
});
