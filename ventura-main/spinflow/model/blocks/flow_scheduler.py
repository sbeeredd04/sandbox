import torch

def get_uniform_step(timestep, total_steps, device='cpu'):
    """
    Get the uniform step size for the given timestep and total steps.
    """
    return torch.tensor([timestep / total_steps], device=device, dtype=torch.float32)

def get_shifted_beta_step(batch_size: int,
                          total_steps: int,
                          device: str | torch.device = "cpu") -> torch.LongTensor:
    """
    Sample diffusion / flow-matching timesteps from a *shifted Beta*
    distribution:

        u            ~  Beta(α=1.5, β=1)
        τ_continuous = total_steps * (1 - u)         ∈ (0, total_steps]
        t_integer    = floor(τ_continuous)           ∈ {0 … total_steps-1}

    Parameters
    ----------
    batch_size : int
        Number of samples to draw.
    total_steps : int
        The maximum number of discrete steps ( “s” in the paper).
    device : str | torch.device, optional
        Where to place the returned tensor.

    Returns
    -------
    torch.LongTensor
        Shape **(batch_size,)**; each entry is an integer timestep with the
        desired shifted-Beta density.
    """
    # 1.  draw u ~ Beta(1.5, 1.0)  on (0,1)
    beta = torch.distributions.Beta(
        torch.tensor(1.5, device=device),
        torch.tensor(1.0, device=device)
    )
    u = beta.sample((batch_size,))                     # (B,)

    # 2.  convert to continuous τ in (0, total_steps]
    tau = total_steps * (1.0 - u)                      # (B,)

    # 3.  integer timestep  t = ⌊τ⌋  ∈ {0,…,total_steps-1}
    timesteps = torch.floor(tau).long()

    # 4.  clamp for numerical safety (rare edge when u≈0)
    timesteps = timesteps.clamp(min=0, max=total_steps - 1)

    return timesteps.to(device)

if __name__ == "__main__":
    # Example usage
    batch_size = 5
    total_steps = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    timesteps = get_shifted_beta_step(batch_size, total_steps, device)
    # Plot timesteps beta plot
    import matplotlib.pyplot as plt
    plt.hist(timesteps.cpu().numpy(), bins=total_steps, alpha=0.5, color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Frequency')
    plt.title('Shifted Beta Timesteps')
    plt.savefig('flow_scheduler.png')