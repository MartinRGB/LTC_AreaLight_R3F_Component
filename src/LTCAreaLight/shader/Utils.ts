export const prefix_vertex = `
    varying vec2 vUv;
    varying vec3 v_pos;

`

export const common_vertex_main = `
    void main()	{
        vUv = uv;
        v_pos = position;
        gl_Position = vec4(position, 1.);
    }
`

export const prefix_frag = `
    #ifdef GL_ES
    precision mediump float;
    #endif

    varying vec3 v_pos;
    varying vec2 vUv;
`