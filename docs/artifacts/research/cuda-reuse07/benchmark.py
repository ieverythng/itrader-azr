"""Synthetic kernel qualification, without importing iTrader or starting Ray."""
import argparse
import copy
import importlib.metadata
import json
import pathlib
import statistics
import time

import torch


def check(actual, expected, *, gradient=False):
    if actual.shape != expected.shape:
        raise AssertionError('shape mismatch')
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=3e-4 if gradient else 2e-4, atol=2e-5)
    return float((actual - expected).abs().max())


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def measured(call, device):
    for _ in range(5):
        call()
    sync(device)
    timings = []
    for block in range(3):
        for _ in range(10):
            sync(device)
            started = time.perf_counter()
            call()
            sync(device)
            timings.append(1000 * (time.perf_counter() - started))
    ordered = sorted(timings)
    return {'median_ms': statistics.median(timings), 'p10_ms': ordered[2],
            'p90_ms': ordered[26], 'sample_count': len(timings), 'samples_ms': timings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    assert torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    report = {'schema': 1, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'device': torch.cuda.get_device_name(),
              'arch_list': torch.cuda.get_arch_list(), 'threads': torch.get_num_threads(),
              'packages': {p: importlib.metadata.version(p) for p in
                           ['torch', 'ray', 'gymnasium', 'numpy']}, 'cases': []}
    for device in ('cpu', 'cuda'):
        a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4).requires_grad_()
        y = a @ torch.eye(4, device=device)
        assert torch.equal(y.detach().cpu(), torch.arange(16, dtype=torch.float32).reshape(4, 4))
        y.sum().backward()
        sync(device)
        assert torch.equal(a.grad.cpu(), torch.ones(4, 4))
    report['arithmetic_autograd'] = 'pass'

    for batch in (1, 128):
        torch.manual_seed(1729)
        cpu = torch.nn.LSTM(5, 156, batch_first=True).float()
        gpu = copy.deepcopy(cpu).cuda()
        xcpu = torch.randn(batch, 60, 5, dtype=torch.float32).requires_grad_()
        xgpu = xcpu.detach().cuda().requires_grad_()
        hcpu = torch.zeros(1, batch, 156)
        ccpu = torch.zeros_like(hcpu)
        hgpu, cgpu = hcpu.cuda(), ccpu.cuda()
        ocpu, (hc, cc) = cpu(xcpu, (hcpu, ccpu))
        ogpu, (hg, cg) = gpu(xgpu, (hgpu, cgpu))
        assert tuple(ocpu.shape) == (batch, 60, 156)
        errors = {'output': check(ogpu, ocpu), 'hidden': check(hg, hc), 'cell': check(cg, cc)}
        lc = ocpu.square().mean() + hc.square().mean() + cc.square().mean()
        lg = ogpu.square().mean() + hg.square().mean() + cg.square().mean()
        lc.backward()
        lg.backward()
        errors['input_gradient'] = check(xgpu.grad, xcpu.grad, gradient=True)
        errors['parameter_gradient'] = max(check(g.grad, c.grad, gradient=True)
                                           for g, c in zip(gpu.parameters(), cpu.parameters()))
        with torch.inference_mode():
            again, states = gpu(xgpu.detach(), (hgpu, cgpu))
        check(again, ogpu)
        check(states[0], hg)
        check(states[1], cg)
        del ocpu, ogpu, lc, lg, hc, cc, hg, cg, again, states

        # A profiler operator check establishes that cuDNN handled the CUDA RNN.
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
            with torch.inference_mode():
                gpu(xgpu.detach(), (hgpu, cgpu))
            torch.cuda.synchronize()
        events = sorted({event.key for event in prof.key_averages()})
        assert any('cudnn_rnn' in key.lower() for key in events), events

        case = {'batch': batch, 'T': 60, 'input': 5, 'hidden': 156,
                'max_absolute_errors': errors,
                'cudnn_operators': [key for key in events if 'cudnn' in key.lower()], 'timings': {}}
        for device, model, x, h, c in [('cpu', cpu, xcpu, hcpu, ccpu),
                                     ('cuda', gpu, xgpu, hgpu, cgpu)]:
            model.eval()
            def inference():
                with torch.inference_mode():
                    model(x.detach(), (h, c))
            model.zero_grad(set_to_none=True)
            x.grad = None
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            inference_times = measured(inference, device)
            inference_memory = {'allocated': torch.cuda.max_memory_allocated(),
                                'reserved': torch.cuda.max_memory_reserved()} if device == 'cuda' else None
            model.train()
            def forward_backward():
                model.zero_grad(set_to_none=True)
                x.grad = None
                out, (hidden, cell) = model(x, (h, c))
                loss = out.square().mean() + hidden.square().mean() + cell.square().mean()
                loss.backward()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            fb_times = measured(forward_backward, device)
            fb_memory = {'allocated': torch.cuda.max_memory_allocated(),
                         'reserved': torch.cuda.max_memory_reserved()} if device == 'cuda' else None
            case['timings'][device] = {'inference': inference_times,
                                       'forward_backward': fb_times,
                                       'inference_cuda_peak_bytes': inference_memory,
                                       'forward_backward_cuda_peak_bytes': fb_memory}
        case['cpu_over_gpu'] = {mode: case['timings']['cpu'][mode]['median_ms'] /
                               case['timings']['cuda'][mode]['median_ms']
                               for mode in ('inference', 'forward_backward')}
        report['cases'].append(case)
        del cpu, gpu, xcpu, xgpu, hcpu, ccpu, hgpu, cgpu
        torch.cuda.empty_cache()
    report['verdict'] = 'synthetic_kernel_qualification_passed'
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({**report, 'cases': [{**c, 'timings': {d: {m: v for m, v in modes.items()
                       if m not in ('inference', 'forward_backward')} | {
                       m: {k: v for k, v in modes[m].items() if k != 'samples_ms'}
                       for m in ('inference', 'forward_backward')}
                       for d, modes in c['timings'].items()}} for c in report['cases']]}, indent=2))


if __name__ == '__main__':
    main()
