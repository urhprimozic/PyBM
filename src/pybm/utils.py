import torch


def torch_interp(x, xp, fp):
    """
    Podobno kot np.interp(x, xp, fp)

    x  - točke, kjer interpoliramo
    xp - x koordinate podatkov (morajo biti sortirane)
    fp - vrednosti v xp
    """
    inds = torch.searchsorted(xp, x)

    inds = torch.clamp(inds, 1, len(xp)-1)

    x0 = xp[inds - 1]
    x1 = xp[inds]

    y0 = fp[inds - 1]
    y1 = fp[inds]

    t = (x - x0) / (x1 - x0)

    return y0 + t * (y1 - y0)