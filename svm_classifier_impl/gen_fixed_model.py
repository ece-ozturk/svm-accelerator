import re
from pathlib import Path

DIM = 784
NSV = 165

def parse_c_array(text, name):
    # matches: double name[] = { ... };
    m = re.search(rf'\bdouble\s+{re.escape(name)}\s*\[\s*\]\s*=\s*\{{(.*?)\}};', text, re.S)
    if not m:
        raise ValueError(f"Could not find array '{name}'")
    body = m.group(1)
    nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', body)
    return [float(x) for x in nums]

def q_feat(v):
    # ap_fixed<8,7>: step = 2^-1 = 0.5
    step = 0.5
    vq = round(v / step) * step
    # saturate to range [-64, 63.5]
    vq = max(-64.0, min(63.5, vq))
    return vq

def q_alpha(v):
    # ap_fixed<8,5>: frac=3 => step=2^-3=0.125, range [-16, 15.875]
    step = 0.125
    vq = round(v / step) * step
    vq = max(-16.0, min(15.875, vq))
    return vq

def q_bias(v):
    # ap_fixed<8,1>: frac=7 => step=2^-7=0.0078125, range [-1, 0.9921875]
    step = 1.0 / 128.0
    vq = round(v / step) * step
    vq = max(-1.0, min(0.9921875, vq))
    return vq

def emit_header(out_path, svs, alphas, bias):
    lines = []
    lines.append('#pragma once')
    lines.append('#include "svm_classifier.h"')
    lines.append('')
    lines.append('static const feat_t svs_q[NSV][DIM] = {')
    for i in range(NSV):
        lines.append('  {')
        row = svs[i*DIM:(i+1)*DIM]
        rowq = [q_feat(v) for v in row]
        # emit 16 per line for readability
        for k in range(0, DIM, 16):
            chunk = rowq[k:k+16]
            lines.append('    ' + ', '.join(f'(feat_t){v:.10g}' for v in chunk) + (',' if k+16 < DIM else ''))
        lines.append('  },')
    lines.append('};')
    lines.append('')
    lines.append('static const alpha_t alphas_q[NSV] = {')
    aq = [q_alpha(v) for v in alphas[:NSV]]
    for k in range(0, NSV, 16):
        chunk = aq[k:k+16]
        tail = ',' if k+16 < NSV else ''
        lines.append('  ' + ', '.join(f'(alpha_t){v:.10g}' for v in chunk) + tail)
    lines.append('};')
    lines.append('')
    lines.append('static const bias_t bias_q[1] = {')
    bq = q_bias(bias[0])
    lines.append(f'  (bias_t){bq:.10g}')
    lines.append('};')
    lines.append('')
    out_path.write_text('\n'.join(lines), encoding='utf-8')

def main():
    # Update these paths to your provided header locations
    svs_h    = Path('c_headers/svs.h').read_text(encoding='utf-8', errors='ignore')
    alphas_h = Path('c_headers/alphas.h').read_text(encoding='utf-8', errors='ignore')
    bias_h   = Path('c_headers/bias.h').read_text(encoding='utf-8', errors='ignore')

    svs    = parse_c_array(svs_h, 'svs')
    alphas = parse_c_array(alphas_h, 'alphas')
    bias   = parse_c_array(bias_h, 'bias')

    if len(svs) < NSV*DIM:
        raise ValueError(f"svs length {len(svs)} < {NSV*DIM}")
    if len(alphas) < NSV:
        raise ValueError(f"alphas length {len(alphas)} < {NSV}")
    if len(bias) < 1:
        raise ValueError("bias length < 1")

    out = Path('svm_model_fixed.h')
    emit_header(out, svs, alphas, bias)
    print(f"Wrote {out} (svs_q[{NSV}][{DIM}], alphas_q[{NSV}], bias_q[1])")

if __name__ == '__main__':
    main()