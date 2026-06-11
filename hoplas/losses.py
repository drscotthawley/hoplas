import torch 


def SIGReg(x, global_step, num_slices=256, chunk_size=32):
    """SIGReg with Epps-Pulley statistic. x is (N, K) tensor.
       Chunked to reduce memory pressure -> More GPU utilization. :-)"""
    with torch.amp.autocast('cuda', enabled=False): # accum in float32
        x = x.float()
        device = x.device
        g = torch.Generator(device=device).manual_seed(global_step)
        A = torch.randn((x.size(1), num_slices), generator=g, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-10)
        t = torch.linspace(-5, 5, 17, device=device)
        exp_f = torch.exp(-0.5 * t**2)
        T_total = torch.tensor(0.0, device=device)    # float32 accumulator
        if chunk_size < 1: chunk_size = num_slices    # < 1 Turns off chunking
        for i in range(0, num_slices, chunk_size):
            x_t = (x @ A[:, i:i+chunk_size]).unsqueeze(2) * t  # (N, chunk, T)
            ecf = (torch.exp(1j * x_t).mean(dim=0)).abs()
            diff = (ecf - exp_f).abs().square().mul(exp_f)
            T_total = T_total + torch.trapz(diff, t, dim=1).sum()
        return T_total