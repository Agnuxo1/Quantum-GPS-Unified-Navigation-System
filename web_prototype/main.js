const canvas = document.getElementById('gl-canvas');
const statusEl = document.getElementById('status');
const debugEl = document.getElementById('debug-info');
const startBtn = document.getElementById('start-btn');

let gl;
let progQuantum, progDisplay;
let vao;
let waveTexA, waveTexB;
let fboA, fboB;
let frame = 0;
let velocity = { x: 0, y: 0 };

// Ping-Pong state
let currentSourceIndex = 0; // 0: Read A, Write B; 1: Read B, Write A

startBtn.addEventListener('click', async () => {
    startBtn.style.display = 'none';
    statusEl.innerText = "Initializing...";

    // Request Permissions (iOS 13+)
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        try {
            const response = await DeviceOrientationEvent.requestPermission();
            if (response === 'granted') {
                initSensors();
                initWebGL();
            } else {
                statusEl.innerText = "Permission Denied";
            }
        } catch (e) {
            statusEl.innerText = "Error: " + e;
        }
    } else {
        // Non-iOS or older devices
        initSensors();
        initWebGL();
    }
});

function initSensors() {
    window.addEventListener('deviceorientation', (e) => {
        // Simple tilt-to-velocity mapping
        // Beta: -180 to 180 (front/back)
        // Gamma: -90 to 90 (left/right)

        // Deadzone
        let bx = e.gamma;
        let by = e.beta;

        if (Math.abs(bx) < 5) bx = 0;
        if (Math.abs(by) < 5) by = 0;

        velocity.x = bx * 2.0;
        velocity.y = by * 2.0;

        debugEl.innerText = `Vel: ${velocity.x.toFixed(1)}, ${velocity.y.toFixed(1)}`;
    });
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vsSource, fsSource) {
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(prog));
        return null;
    }
    return prog;
}

function createTexture(gl, width, height) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // RGBA32F for high precision
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
}

function initWebGL() {
    gl = canvas.getContext('webgl2');
    if (!gl) {
        statusEl.innerText = "WebGL 2 Not Supported";
        return;
    }

    // Extensions
    gl.getExtension('EXT_color_buffer_float'); // Crucial for RGBA32F FBOs

    // Resize
    canvas.width = 512;
    canvas.height = 512;
    gl.viewport(0, 0, 512, 512);

    // Shaders
    progQuantum = createProgram(gl, SHADERS.vs, SHADERS.quantum_fs);
    progDisplay = createProgram(gl, SHADERS.vs, SHADERS.display_fs);

    // Buffers
    const quad = new Float32Array([
        -1, -1,
        1, -1,
        -1, 1,
        1, 1,
    ]);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const loc = gl.getAttribLocation(progQuantum, 'in_vert');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    // Textures & FBOs
    waveTexA = createTexture(gl, 512, 512);
    waveTexB = createTexture(gl, 512, 512);

    fboA = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboA);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, waveTexA, 0);

    fboB = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, waveTexB, 0);

    // Init Wavefunction (Center Gaussian)
    const initialData = new Float32Array(512 * 512 * 4);
    for (let y = 0; y < 512; y++) {
        for (let x = 0; x < 512; x++) {
            const dx = x - 256;
            const dy = y - 256;
            if (dx * dx + dy * dy < 20 * 20) {
                const idx = (y * 512 + x) * 4;
                initialData[idx] = 1.0; // R channel
            }
        }
    }
    gl.bindTexture(gl.TEXTURE_2D, waveTexA);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 512, 512, gl.RGBA, gl.FLOAT, initialData);

    statusEl.innerText = "System Active";
    requestAnimationFrame(loop);
}

function loop() {
    // 1. Quantum Step
    gl.useProgram(progQuantum);

    // Bind Input Texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentSourceIndex === 0 ? waveTexA : waveTexB);
    gl.uniform1i(gl.getUniformLocation(progQuantum, 'u_wave'), 0);

    // Uniforms
    gl.uniform2f(gl.getUniformLocation(progQuantum, 'u_velocity'), velocity.x, velocity.y);
    gl.uniform1f(gl.getUniformLocation(progQuantum, 'u_dt'), 0.05);
    gl.uniform1f(gl.getUniformLocation(progQuantum, 'u_diffusion'), 0.001);
    gl.uniform2f(gl.getUniformLocation(progQuantum, 'u_res'), 512, 512);

    // Bind Output FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, currentSourceIndex === 0 ? fboB : fboA);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // 2. Display Step
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Screen
    gl.useProgram(progDisplay);

    // Bind the texture we just wrote to
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentSourceIndex === 0 ? waveTexB : waveTexA);
    gl.uniform1i(gl.getUniformLocation(progDisplay, 'u_wave'), 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Swap
    currentSourceIndex = 1 - currentSourceIndex;

    requestAnimationFrame(loop);
}
