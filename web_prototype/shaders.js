const SHADERS = {
    vs: `#version 300 es
    in vec2 in_vert;
    out vec2 v_uv;
    void main() {
        v_uv = in_vert * 0.5 + 0.5;
        gl_Position = vec4(in_vert, 0.0, 1.0);
    }`,

    quantum_fs: `#version 300 es
    precision highp float;
    
    uniform sampler2D u_wave;
    uniform sampler2D u_obstacle;
    uniform vec2 u_velocity;
    uniform float u_dt;
    uniform float u_diffusion;
    uniform vec2 u_res;
    
    in vec2 v_uv;
    out vec4 f_color;
    
    void main() {
        vec2 pixel = 1.0 / u_res;
        vec4 c = texture(u_wave, v_uv);
        // float obs = texture(u_obstacle, v_uv).r; // Disabled for now
        
        // Advection (move with velocity)
        vec2 pos = v_uv - u_velocity * u_dt * pixel;
        vec4 val = texture(u_wave, pos);
        
        // Diffusion (spread out)
        vec4 sum = val;
        sum += texture(u_wave, pos + vec2(pixel.x, 0.0));
        sum += texture(u_wave, pos - vec2(pixel.x, 0.0));
        sum += texture(u_wave, pos + vec2(0.0, pixel.y));
        sum += texture(u_wave, pos - vec2(0.0, pixel.y));
        
        f_color = mix(val, sum / 5.0, u_diffusion);
        
        // Decay
        f_color *= 0.995;
    }`,

    display_fs: `#version 300 es
    precision highp float;
    
    uniform sampler2D u_wave;
    uniform vec2 u_res;
    
    in vec2 v_uv;
    out vec4 f_color;
    
    vec3 plasma(float t) {
        const vec3 c0 = vec3(0.0, 0.0, 0.0);
        const vec3 c1 = vec3(0.2, 0.0, 0.4);
        const vec3 c2 = vec3(0.0, 0.5, 1.0);
        const vec3 c3 = vec3(0.0, 1.0, 0.5);
        const vec3 c4 = vec3(1.0, 1.0, 0.0);
        
        t = clamp(t, 0.0, 1.0);
        if (t < 0.25) return mix(c0, c1, t * 4.0);
        if (t < 0.5)  return mix(c1, c2, (t - 0.25) * 4.0);
        if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
        return mix(c3, c4, (t - 0.75) * 4.0);
    }
    
    void main() {
        vec4 wave = texture(u_wave, v_uv);
        float prob = wave.r;
        
        vec3 color = vec3(0.05, 0.05, 0.1); // Background
        
        // Grid lines
        vec2 grid = fract(v_uv * 20.0);
        if (grid.x < 0.05 || grid.y < 0.05) color += 0.1;
        
        if (prob > 0.001) {
            color += plasma(prob * 5.0);
        }
        
        f_color = vec4(color, 1.0);
    }`
};
