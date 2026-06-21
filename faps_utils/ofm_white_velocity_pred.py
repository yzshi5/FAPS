import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faps_utils.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples
# independent coupling 
#from torchcfm.optimal_transport import OTPlanSampler


import time


class WhiteNoiseGenerator:
    """
    White noise generator - samples from standard normal distribution.
    Replaces Gaussian Process prior with simple white noise.
    """
    
    def __init__(self, alpha=None, device='cpu', dims=None, scale=None):
        """
        Args:
            alpha: ignored (kept for compatibility)
            device: device to generate noise on
            dims: spatial dimensions [n_x, n_y, n_t]
            scale: ignored (kept for compatibility)
        """
        self.device = device
        self.dims = dims if dims is not None else []
    
    def sample_from_prior(self, n_samples=1, n_channels=1):
        """
        Sample white noise from standard normal distribution.
        
        Args:
            n_samples: number of samples
            n_channels: number of channels
            
        Returns:
            Tensor of shape [n_samples, n_channels, *dims] with white noise
        """
        shape = [n_samples, n_channels] + list(self.dims)
        return torch.randn(*shape, device=self.device)
    
    def sample(self, dims, n_samples=1, n_channels=1):
        """
        Sample white noise with specified dimensions.
        
        Args:
            dims: spatial dimensions [n_x, n_y, n_t]
            n_samples: number of samples
            n_channels: number of channels
            
        Returns:
            Tensor of shape [n_samples, n_channels, *dims] with white noise
        """
        shape = [n_samples, n_channels] + list(dims)
        return torch.randn(*shape, device=self.device)

class OFMModel:
    def __init__(
        self,
        model,
        alpha=None,
        scale=None,
        sigma_min=1e-4,
        device='cpu',
        dtype=torch.float32,
        dims=None,
        noise_scale=1.0,
        t_eps=0.05,
        sample_dtype=torch.float32,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        # Use white noise generator instead of Gaussian Process
        self.gp = WhiteNoiseGenerator(alpha=alpha, device=device, dims=dims, scale=scale)
        #self.ot_sampler = OTPlanSampler(method="exact")
        self.sigma_min = sigma_min
        self.t_mu = -0.8
        self.t_sigma = 0.8
        self.t_eps = t_eps # clip of (1 -t) in division
        self.noise_scale = noise_scale
        self.sample_dtype = sample_dtype

    def sample_t(self, n: int, device=None):
        """ 
        Sample diffusion times using a logistic-normal distribution such that
        logit(t) ~ N(mu, sigma^2). Defaults to mu=-0.8, sigma=0.8.
        """
        if device is None:
            device = self.device

        """
        z = torch.randn(n, device=device) * self.t_sigma + self.t_mu
        return torch.sigmoid(z)
        """    
        return torch.rand(n, device=device)


    def sample_gp_noise(self, x_data, x_conds):
        # sample GP noise with OT 
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        
        # GP noise : [batch_size, n_channels, *dims]
        #x_0 = self.gp.sample_from_prior(query_points, dims, n_samples=batch_size, n_channels=n_channels) 
        x_0 = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels) 
        
        #x_0, x_data, _, x_conds = self.ot_sampler.sample_plan_with_labels(x_0, x_data, y1=x_conds)
        
        return x_0, x_data, x_conds
        
    def simulate(self, t, x_0, x_data):
        # t: [batch_size,]
        # x_data: [batch_size, n_channels, *dims]
        # samples from p_t(x | x_data)
        
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]

        # Sample from prior GP
        # we should define a second Gaussian kernel for the GP noise, here, we set it the same as p_0(x) 
        #query_points = make_grid(dims)
        #noise = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels) # GP noise : [batch_size, n_channels, *dims]
        noise = self.gp.sample_from_prior(n_samples=batch_size, n_channels=n_channels)

        
        mu = t * x_data + (1 - t) * x_0
        samples = mu + self.sigma_min * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, x0, x1, t):
        # computes v_t(x_noisy | x_data)
        # x_data, x_noisy: (batch_size, n_channels, *dims)

        return (x1 - x0) * ((1 - t) / (1 - t).clamp_min(self.t_eps))

    def train(self, train_loader, optimizer, epochs,
                scheduler=None, test_loader=None, eval_int=0,
                save_int=0, generate=False, save_path=None, saved_model=False,
                loss_filename='loss.pdf', loss_log_filename=None):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)
        loss_log_path = save_path / loss_log_filename if loss_log_filename else None

        model = self.model
        device = self.device
        dtype = self.dtype

        if loss_log_path is not None:
            loss_log_path.write_text("epoch,train_loss,test_loss\n", encoding="utf-8")

        first = True
        for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0
            te_loss_value = None

            for batch, init_conds in train_loader:
                batch, init_conds = batch.to(device), init_conds.to(device)
                batch_size = batch.shape[0]

                if first:
                    self.n_channels = batch.shape[1]
                    self.train_dims = batch.shape[2:]
                    first = False
                    
                # GP noise with OT reorder
                x_0, x_data, conds = self.sample_gp_noise(batch, init_conds)
        
                # t ~ logistic-normal in (0, 1)
                t = self.sample_t(batch_size, device=device)
                t = reshape_for_batchwise(t, len(x_0.shape[1:]))

                # Simluate p_t(x | x_1)
                x_t = self.simulate(t, x_0, x_data)
                # Get conditional vector fields
                target = self.get_conditional_fields(x_0, x_data, t)

                x_t = x_t.to(device)
                target = target.to(device) # v loss        

                # Get model output
                #print('t before the model :{}'.format(t))
                # conds shape : [batch_size, n_conds]
                model_out_v = model(t.squeeze(), x_t, conds)  # direct velocity prediction
                

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = torch.mean((model_out_v - target)**2 ) 
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()


            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            ##### BOOKKEEPING
            if saved_model == True:
                if ep % save_int == 0:
                    torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')

            if loss_log_path is not None:
                te_loss_str = "" if te_loss_value is None else f"{te_loss_value:.10g}"
                with loss_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{ep},{tr_loss:.10g},{te_loss_str}\n")

            if loss_filename is not None:
                loss_path = save_path / loss_filename
                if evaluate:
                    plot_loss_curve(tr_losses, loss_path, te_loss=te_losses, te_epochs=eval_eps)
                else:
                    plot_loss_curve(tr_losses, loss_path)



    @torch.no_grad()
    def sample(
        self,
        dims,
        conds=None,
        n_channels=1,
        n_samples=1,
        n_eval=50,
        method='heun',
        sample_dtype=None,
    ):
        """
        Deterministic sampler following Euler/Heun integration of the learned flow.
        """
        if conds is None:
            raise ValueError("conds must be provided when calling sample.")

        work_dtype = sample_dtype if sample_dtype is not None else self.sample_dtype
        conds = conds.to(self.device, dtype=work_dtype)
        time_steps = torch.linspace(0, 1, n_eval+1, device=self.device, dtype=work_dtype)

        # t = 0, x_t shape [batch_size, n_channels, *dims]
        x_t = self.noise_scale * self.gp.sample(dims, n_samples=n_samples, n_channels=n_channels)
        x_t = x_t.to(self.device, dtype=work_dtype)

        if method == 'euler':
            stepper = self._euler_step
        elif method == 'heun':
            stepper = self._heun_step
        else:
            raise ValueError(f"Unsupported method: {method}")

        for i in range(n_eval-1):
            t = time_steps[i]
            t_next = time_steps[i+1]
            x_t = stepper(x_t, t, t_next, conds)

        # last step euler 
        x_t = self._euler_step(x_t, time_steps[-2], time_steps[-1], conds)

        return x_t

    @torch.no_grad()
    def sample_with_odeint(
        self,
        dims,
        conds=None,
        n_channels=1,
        n_samples=1,
        n_eval=20,
        rtol=1e-5,
        atol=1e-5,
        method='dopri5',
        sample_dtype=None,
    ):
        """
        ODE solver-based sampler (e.g., dopri5) with velocity prediction.
        """
        if conds is None:
            raise ValueError("conds must be provided when calling sample_with_odeint.")

        work_dtype = sample_dtype if sample_dtype is not None else self.sample_dtype
        conds = conds.to(self.device, dtype=work_dtype)
        t_eval = torch.linspace(0, 1, n_eval+1, device=self.device, dtype=work_dtype)
        t_odeint = torch.Tensor([t_eval[0], t_eval[-2]])

        x0 = self.noise_scale * self.gp.sample(dims, n_samples=n_samples, n_channels=n_channels)
        x0 = x0.to(self.device, dtype=work_dtype)

        def vector_field(t_scalar, x_state):
            return self._ode_rhs_vpred(t_scalar, x_state, conds)

        out = odeint(
            vector_field,
            x0,
            t_odeint,
            method=method,
            rtol=rtol,
            atol=atol,
        )[-1]

        # last step euler
        out = self._euler_step(out, t_eval[-2], t_eval[-1], conds)
        return out

    def _reshape_like_state(self, tensor_1d, state):
        tensor_1d = tensor_1d.to(state.device, dtype=state.dtype)
        return reshape_for_batchwise(tensor_1d, state.ndim - 1)

    def _expand_time(self, t, batch_size, device, dtype):
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                return torch.full((batch_size,), t.item(), device=device, dtype=dtype)
            if t.shape[0] == batch_size:
                return t.to(device=device, dtype=dtype)
            if t.numel() == 1:
                return t.to(device=device, dtype=dtype).repeat(batch_size)
            raise ValueError(f"Time tensor with shape {t.shape} cannot broadcast to batch size {batch_size}.")
        return torch.full((batch_size,), float(t), device=device, dtype=dtype)

    @torch.no_grad()
    def _forward_sample(self, x_t, t, conds):
        bsz = x_t.shape[0]
        t_vec = self._expand_time(t, bsz, x_t.device, x_t.dtype)
        conds = conds.to(x_t.device, dtype=x_t.dtype)
        v_cond = self.model(t_vec.float(), x_t.float(), conds.float()).to(x_t.dtype)
        return v_cond

    def _ode_rhs_vpred(self, t_scalar, x_state, conds):
        t_vec = self._expand_time(t_scalar, x_state.shape[0], x_state.device, x_state.dtype)
        conds = conds.to(x_state.device, dtype=x_state.dtype)
        v = self.model(t_vec.float(), x_state.float(), conds.float()).to(x_state.dtype)
        return v

    @torch.no_grad()
    def _euler_step(self, x_t, t, t_next, conds):
        v_pred = self._forward_sample(x_t, t, conds)
        x_next = x_t + (t_next - t) * v_pred
        return x_next 


    @torch.no_grad()
    def _heun_step(self, x_t, t, t_next, conds):
        v_pred_t = self._forward_sample(x_t, t, conds)
        x_next_euler = x_t + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(x_next_euler, t_next, conds)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        x_next = x_t + (t_next - t) * v_pred
        return x_next 



#### For evaluatoin ###         
"""
from faps_utils.unet_ofm import UNet_cond    
def _default_device():
    return "cuda:1" if torch.cuda.is_available() else "cpu"


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Sample from latent OFM model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--in-channels", type=int, default=1, help="Number of channels (e.g. 1 for the requested shape).")
    parser.add_argument("--depth", type=int, default=32, help="Depth dimension of the sample.")
    parser.add_argument("--height", type=int, default=16, help="Height dimension of the sample.")
    parser.add_argument("--width", type=int, default=16, help="Width dimension of the sample.")
    parser.add_argument("--num-conds", type=int, default=4, help="Number of conditioning scalars.")
    parser.add_argument("--cond-value", type=float, default=0.0, help="Constant value to fill condition vector.")
    parser.add_argument("--hidden-channels", type=int, default=32, help="Hidden channels within the UNet backbone.")
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--attention-res", type=str, default="16")
    parser.add_argument("--channel-mult", type=str, default="", help="Comma separated multipliers for UNet channels.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to a model checkpoint.")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE steps for sampling.")
    parser.add_argument("--method", choices=["euler", "heun"], default="heun", help="Integration scheme.")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Scale of the initial noise.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Device to run sampling on.")
    parser.add_argument("--output", type=str, default="", help="Optional path to save generated samples (torch.save).")
    return parser


def _maybe_parse_channel_mult(spec: str):
    if not spec:
        return None
    return tuple(int(token.strip()) for token in spec.split(",") if token.strip())


def _load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


def _build_model(args, device):
    channel_mult = _maybe_parse_channel_mult(args.channel_mult)
    net = UNet_cond(
        dims=[args.in_channels, args.depth, args.height, args.width],
        hidden_channels=args.hidden_channels,
        conds_channels=args.num_conds,
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        attention_res=args.attention_res,
        channel_mult=channel_mult,
        in_channels=args.in_channels,
    )
    return net.to(device=device, dtype=torch.float32)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    # print the configuration
    print(args)

    device = torch.device(args.device)
    spatial_dims = (args.depth, args.height, args.width)

    model = _build_model(args, device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, device)
    model.eval()

    flow = OFMModel(
        model=model,
        device=device,
        dtype=torch.float32,
        dims=spatial_dims,
        gen_steps=args.steps,
        gen_method=args.method,
        noise_scale=args.noise_scale,
    )
    flow.n_channels = args.in_channels
    flow.train_dims = spatial_dims

    conds = torch.full(
        (args.batch_size, args.num_conds),
        args.cond_value,
        device=device,
        dtype=torch.float32,
    )

    samples = flow.sample(
        dims=spatial_dims,
        conds=conds,
        n_channels=args.in_channels,
        n_samples=args.batch_size,
        n_eval=args.steps,
        method=args.method,
    )

    print(f"Generated samples shape: {tuple(samples.shape)}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples.cpu(), out_path)
        print(f"Saved samples to {out_path.resolve()}")


if __name__ == "__main__":
    main()
"""
