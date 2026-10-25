import bevy_pbr::prepass::io as prepass_io;
import bevy_pbr::render::pbr_prepass_functions;
import bevy_pbr::render::pbr_functions;

struct ReceiverIdentity { value: vec4<u32> }
@group(1) @binding(0) var<uniform> receiver: ReceiverIdentity;

struct ReceiverOutput {
    @location(0) identity: vec4<u32>,
    @if(UNCLIPPED_DEPTH_ORTHO_EMULATION)
    @builtin(frag_depth) depth: f32,
}

fn pack_geometric_normal(position: vec3<f32>) -> u32 {
    let dx = dpdx(position);
    let dy = dpdy(position);
    let scale = max(max(max(abs(dx.x), abs(dx.y)), abs(dx.z)),
                    max(max(abs(dy.x), abs(dy.y)), abs(dy.z)));
    if (!(scale > 0.0) || scale > 3.402823e38) {
        return 0u;
    }
    let normal = cross(dx / scale, dy / scale);
    let magnitude = abs(normal.x) + abs(normal.y) + abs(normal.z);
    if (!(magnitude > 0.0)) {
        return 0u;
    }
    var oct = normal.xy / magnitude;
    if (normal.z < 0.0) {
        oct = (vec2<f32>(1.0) - abs(oct.yx))
            * select(vec2<f32>(-1.0), vec2<f32>(1.0), oct >= vec2<f32>(0.0));
    }
    let quantized = vec2<u32>(round((oct * 0.5 + vec2<f32>(0.5)) * 32767.0));
    // Bit 31 marks eligibility; bits 0..29 hold the octahedral normal.
    return 0x80000000u | quantized.x | (quantized.y << 15u);
}

@fragment
fn fragment(in: prepass_io::VertexOutput) -> ReceiverOutput {
    let packed_normal = pack_geometric_normal(in.world_position.xyz);
    @if(VISIBILITY_RANGE_DITHER)
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);

    pbr_prepass_functions::prepass_sample_color_and_alpha_discard(in);
    var result: ReceiverOutput;
    result.identity = receiver.value;
    result.identity.z = packed_normal;
    @if(UNCLIPPED_DEPTH_ORTHO_EMULATION)
    result.depth = in.unclipped_depth;
    return result;
}
