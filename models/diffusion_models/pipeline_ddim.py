
# modified from  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py

from typing import List, Optional, Tuple, Union

import torch

from diffusers.schedulers import DDIMScheduler

from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

# from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from util import ProGenPath, instantiate_from_config

import torch.nn as nn
import time



class DDIMPipeline1D(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        # postprocessing here # TODO

        return image

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class DDIMSchedulerCustom(DDIMScheduler):
    def get_alpha_beta(self, timestep):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod # 1

        beta_prod_t = 1 - alpha_prod_t
        return alpha_prod_t, alpha_prod_t_prev, beta_prod_t

    def forward_step(self, timestep, x_t_prev):
        # Adding noise from x_(t-1) in the forward diffusion process
        alpha_prod_t, alpha_prod_t_prev, beta_prod_t = self.get_alpha_beta(timestep)
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        x_t = alpha_t.sqrt() * x_t_prev + (1 - alpha_t).sqrt() * noise_like(x_t_prev.shape, x_t_prev.device, False)
        # Reference
        # img = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(img.shape, device, False)
        return x_t

    def get_noise_scale(self, timestep):
        beta_prod_t = 1 - self.alphas_cumprod[timestep]
        noise_scale = beta_prod_t ** (0.5)
        return noise_scale

    def get_original_sample(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        # eta: float = 0.0,
        # use_clipped_model_output: bool = False,
        # generator=None,
        # variance_noise: Optional[torch.FloatTensor] = None,
        # return_dict: bool = True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            # eta (`float`):
            #     The weight of noise for added noise in diffusion step.
            # use_clipped_model_output (`bool`, defaults to `False`):
            #     If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
            #     because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
            #     clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
            #     `use_clipped_model_output` has no effect.
            # generator (`torch.Generator`, *optional*):
            #     A random number generator.
            # variance_noise (`torch.FloatTensor`):
            #     Alternative to generating noise with `generator` by directly providing the noise for the variance
            #     itself. Useful for methods such as [`CycleDiffusion`].
            # return_dict (`bool`, *optional*, defaults to `True`):
            #     Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            original_sample (`torch.FloatTensor`)
            # [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            #     If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
            #     tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        alpha_prod_t, alpha_prod_t_prev, beta_prod_t = self.get_alpha_beta(timestep)
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # # 4. Clip or threshold "predicted x_0"
        # if self.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )
        return pred_original_sample


class neg_reward_loss:
    def __call__(self, reward):
        return -1. * reward.sum()

class GuidedDDIMPipeline1D(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    def __init__(
            self, unet, scheduler, 
            # guidance
            classifier_model_config=None,
            encdec=None,
            loss_type="l2", # "l2" or "neg_reward"
            guidance_scale=1.0,
            # do_sample=False,
            # top_k = 50,
            # top_p = 1.0,
            # eos_token = "<|eos|>",
            ):
        super().__init__()
        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMSchedulerCustom.from_config(scheduler.config)
        self.register_modules(unet=unet, scheduler=scheduler)
        self.classifier_model = None
        if classifier_model_config is not None:
            self.classifier_model = instantiate_from_config(classifier_model_config)
            self.classifier_type = self.classifier_model.classifier_type
            self.guidance_type = self.classifier_model.guidance_type
        self.encdec = encdec
        self.softmax = torch.nn.Softmax(dim=-1)
        self.guidance_scale = guidance_scale
        if loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            # self.loss = neg_reward_loss()
            self.loss = nn.CrossEntropyLoss()
        if self.encdec is not None:
            self.decoder_a_id = self.encdec.dec_tokenizer.token_to_id("A") # 5 

    def logits_to_prob(self, logits):
        logits = torch.stack(logits, dim=0)
        logits = logits.permute(1, 0, 2) # (batch_size, seq_len, vocab_size)
        vocab_prob = self.softmax(logits)
        # valid_prob = vocab_prob[:,:,5:9] / torch.sum(vocab_prob[:,:,5:9], dim=-1, keepdim=True) # A, C, G, U, [5, 6, 7, 8]
        valid_prob = vocab_prob[:,:,self.decoder_a_id:self.decoder_a_id+4] / torch.sum(vocab_prob[:,:,self.decoder_a_id:self.decoder_a_id+4], dim=-1, keepdim=True) # A, C, G, U
        return valid_prob
    
    def postprocess_seqs(self, seqs, valid_prob, update_seq=True):
        n_seqs = []
        n_probs = []
        for si, seq in enumerate(seqs):
            non_1_2_pos = torch.ones(len(seq))
            n_seq = ""
            for ti, token in enumerate(seq):
                if token in ["1", "2"]: # Hopefully for each seq in batch, "1" and "2" are present in the same position
                    non_1_2_pos[ti] = 0 # remove "1" and "2"
                    continue
                if update_seq:
                    token = torch.argmax(valid_prob[si][ti])
                n_seq += token
            n_seqs.append(n_seq)   
            n_probs.append(valid_prob[si][non_1_2_pos.bool(), :]) 
        n_probs = torch.stack(n_probs, dim=0)
        return n_seqs, n_probs

    def prepare_input_for_reward_model(self, logits, seqs, update_seq=False): #TODO: fit for the variable length of seqs
        valid_prob = self.logits_to_prob(logits)
        if not update_seq:
            seqs, valid_prob = self.postprocess_seqs(seqs, valid_prob, update_seq=update_seq)
            # get reward and gradient with regard to the one-hot encoding of the sequence
            reward_in = self.reward_model.one_hot_all_motif(seqs).to(self.reward_model.device)
        else:
            rm_s_1 = False 
            n_seqs = []
            for i, seq in enumerate(seqs):
                if seq.startswith("1"):
                    rm_s_1 = True
                if rm_s_1:
                    n_seqs.append(seq[1:]) # TODO: n_seqs
            if rm_s_1:
                valid_prob = valid_prob[:, 1 :,:]
                seqs = n_seqs
            reward_in = nn.functional.one_hot(valid_prob.argmax(dim=-1), num_classes=4).float() #TODO: 
            reward_in.to(self.reward_model.device)
        return reward_in, valid_prob, seqs    
    # A. Support seq-level reward model
    def get_gradient(self, t, image, target, max_seq_len=150, update_seq=False, verbose=False, **decoder_sample_kwargs):
        with torch.enable_grad():
            start_time = time.time()
            x_in = image.detach().requires_grad_(True)
            model_output = self.unet(x_in, t).sample 
            if verbose:
                print("diffusion model: ", time.time() - start_time)
            start_time = time.time()
            # get z_0_hat
            original_sample = self.scheduler.get_original_sample(model_output, t, x_in)
            if verbose:
                print("get estimation of z0: ", time.time() - start_time)
            start_time = time.time()            
            # decode (generate)
            logits, seqs = self.encdec.generate(
                original_sample, max_seq_len,
                **decoder_sample_kwargs
                # do_sample=self.do_sample, top_k=self.top_k, top_p=self.top_p,
                # eos_token=self.decoder_generation_eos_token,
                )  #TODO: decide the eos token to  [Time-Consuming]
            if verbose:
                print("decoder generation: ", time.time() - start_time)
            start_time = time.time()  
            # logits: # args.max_seq_len * (batch_size, vocab_size)
            reward_in, valid_prob, seqs = self.prepare_input_for_reward_model(logits, seqs, update_seq=update_seq) #TODO: suitable for the variable length of seqs
            reward_in.requires_grad_(True)
            if verbose:
                print("prepare input for reward model: ", time.time() - start_time)
            start_time = time.time()  
            torch.backends.cudnn.enabled=False #TODO: check if this is correct
            reward = self.reward_model(reward_in)
            loss = self.loss(target * torch.ones(reward.shape, device=reward.device), reward)
            print("loss:", loss)
            reward_gradient = torch.autograd.grad(loss, reward_in)[0]
            torch.backends.cudnn.enabled=True #TODO: check if this is correct
            if verbose:
                print("get gradient of the seq: ", time.time() - start_time)
            start_time = time.time()  
            # with torch.autograd.detect_anomaly():
            # copy the gradient back to the seq logits
            gradient = torch.autograd.grad(valid_prob, x_in, grad_outputs=reward_gradient)[0] # [Time-Consuming]
            if verbose:
                print("get gradient of the zt: ", time.time() - start_time)
            # pred_noise = self.model(x_in, t, classes=None)
            # noise_svd = extract(
            #     self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape)
            # uncond_score = -1 * pred_noise * 1 / noise_svd
            # x_0_hat = (x_in + h_t * uncond_score) / alpha_t
            # # value = torch.matmul(x_0_hat, self.g.view(-1, 1))
            # value = torch.sum(x_0_hat * self.g, dim=1, keepdim=True)
            # gradient = torch.autograd.grad(value.sum(), x_in)[0]
        return gradient, model_output

    # B. Support latent-level reward model
    def get_gradient_latent(self, t, image, target_class, verbose=False):
        with torch.enable_grad():
            start_time = time.time()
            x_in = image.detach().requires_grad_(True)
            model_output = self.unet(x_in, t).sample 
            if verbose:
                print("diffusion model: ", time.time() - start_time)
            start_time = time.time()
            # get z_0_hat
            original_sample = self.scheduler.get_original_sample(model_output, t, x_in)
            if verbose:
                print("get estimation of z0: ", time.time() - start_time)
            reward_in = original_sample
            start_time = time.time()  
            # torch.backends.cudnn.enabled=False #TODO: check if this is correct
            # reward = self.reward_model(reward_in)
            class_logits = self.classifier_model(reward_in)

            # loss = self.loss(target * torch.ones(reward.shape, device=reward.device), reward)
            target = torch.tensor([target_class]*x_in.size(0), device=x_in.device)
            loss = self.loss(class_logits, target)

            print("loss:", loss.item())
            if verbose:
                print("passing through classifier model and calculating loss ", time.time() - start_time)
                print("loss:", loss)
            start_time = time.time()  
            gradient = torch.autograd.grad(loss, x_in)[0]
            # reward_gradient = torch.autograd.grad(loss, reward_in)[0]
            # torch.backends.cudnn.enabled=True #TODO: check if this is correct
            # with torch.autograd.detect_anomaly():
            if verbose:
                print("get gradient of the zt: ", time.time() - start_time)
        return gradient, model_output

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        # return_dict: bool = True,
        num_query_tokens = 16, # [FIXED] Should be changed according to the model
        guidance = False, 
        target_class = None,
        update_seq = False,
        max_seq_len = 150,
        verbose=False,
        recurrence_step = 1,
        guidance_start_step = 0,
        guidance_stop_step = None,
        **decoder_sample_kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

                
            guidance ('bool', *optional*, defaults to 'False'):
                Whether to add (gradient) guidance in the denoising process. 
                guidance type can be one of ['TE', 'BS']. 'TE' means translation efficiency while 'BS' means (RNA-protein) binding score.
            target ('torch.Tensor', *optional*, defaults to 'None'):
                target reward score for the guidance.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Sample gaussian noise to begin loop
        if "sample_size" in self.unet.config: 
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        
        else: # [TransformerDenoiser]
            image_shape = (
                batch_size,
                num_query_tokens,
                self.unet.config.in_channels
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if guidance_stop_step is None:
            guidance_stop_step = num_inference_steps
        else:
            guidance_stop_step = min(guidance_stop_step, num_inference_steps)
        for denoise_step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # TODO: change the device of t to cuda
            # 1.0 add guidance
            if guidance and self.classifier_model is not None and denoise_step >= guidance_start_step and denoise_step < guidance_stop_step:
                # Reference from Universal Guidance
                # get gradient
                for i in range(recurrence_step):
                    if i == 0:
                        image_t = image # z_t (image)
                    else: 
                        image_t = self.scheduler.forward_step(t, image) # build from z_t-1 (image)

                    if self.guidance_type == "seq":
                        gradient, model_output = self.get_gradient(t, image_t, target, update_seq=update_seq, max_seq_len=max_seq_len, verbose=verbose, **decoder_sample_kwargs)
                    else: # "latent"
                        gradient, model_output = self.get_gradient_latent(t, image_t, target_class, verbose=verbose) ## model_output detach() ？
                    # add gradient to model_output
                    model_output = model_output + self.guidance_scale * self.scheduler.get_noise_scale(t) *  gradient
                    image = self.scheduler.step(
                        model_output, t, image_t, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                    ).prev_sample  # z_t-1
                # print("model output norm: ", torch.norm(model_output))
            else:
                # 1. predict noise model_output
                model_output = self.unet(image, t).sample # noise prediction
                # print("model output norm: ", torch.norm(model_output))
                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t, image, eta=eta, 
                    use_clipped_model_output=use_clipped_model_output, generator=generator
                ).prev_sample
            # print(torch.norm(image))

        # postprocessing here # TODO

        return image
    





class DDIMSchedulerCustomMutiple(DDIMSchedulerCustom):
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        branch_size: int = 1,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:

                ## branch out here
                prev_sample = torch.repeat_interleave(prev_sample, repeats=branch_size, dim=0)  # [branch_size * batch_size, *image_shape]




                variance_noise = randn_tensor(
                    prev_sample.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    

    def get_random_noise_scale(self, timestep, eta):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        return std_dev_t





class SearchDDIMPipeline1D(GuidedDDIMPipeline1D):
    def __init__(
        self,
        unet,
        scheduler,
        classifier_model_config=None,
        encdec=None,
        loss_type="l2",
        guidance_scale=1.0,
        active_size=1,
        branch_size=1,
    ):
       
        # Store the additional parameters for the child class.
        self.active_size = active_size
        self.branch_size = branch_size


        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMSchedulerCustomMutiple.from_config(scheduler.config)


        self.register_modules(unet=unet, scheduler=scheduler)
        self.classifier_model = None
        if classifier_model_config is not None:
            self.classifier_model = instantiate_from_config(classifier_model_config)
            self.classifier_type = self.classifier_model.classifier_type
            self.guidance_type = self.classifier_model.guidance_type
        self.encdec = encdec
        self.softmax = torch.nn.Softmax(dim=-1)
        self.guidance_scale = guidance_scale
        if loss_type == "l2":
            self.loss = nn.MSELoss()
            self.loss_func = nn.MSELoss(reduction='none')
        else:
            # self.loss = neg_reward_loss()
            self.loss = nn.CrossEntropyLoss()
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        if self.encdec is not None:
            self.decoder_a_id = self.encdec.dec_tokenizer.token_to_id("A") # 5 


    def evaluate_then_select(self, batch_size, t, image, target_class):

        model_output = self.unet(image, t).sample
        original_sample = self.scheduler.get_original_sample(model_output, t, image)

        class_logits = self.classifier_model(original_sample)
        target = torch.tensor([target_class]*image.size(0), device=image.device)
        loss = self.loss_func(class_logits, target)

        print("loss shape: ", loss.shape)
        # print("loss: ", loss)

        ## select the top active_size
        loss = loss.view(batch_size, -1)
        topk_loss, topk_idx = torch.topk(loss, self.active_size, dim=1, largest=False)

        image = image.view(batch_size, -1, *image.shape[1:])
        image = image[torch.arange(batch_size).unsqueeze(1), topk_idx]
        image = image.view(-1, *image.shape[2:])

        return image, topk_loss
    

    def select_clean(self, batch_size, top_size, clean_image, target_class):
        class_logits = self.classifier_model(clean_image)
        target = torch.tensor([target_class]*clean_image.size(0), device=clean_image.device)
        loss = self.loss_func(class_logits, target)

        ## select the top active_size
        loss = loss.view(batch_size, -1)
        top_loss, top_idx = torch.topk(loss, top_size, dim=1, largest=False)

        clean_image = clean_image.view(batch_size, -1, *clean_image.shape[1:])
        clean_image = clean_image[torch.arange(batch_size).unsqueeze(1), top_idx]
        clean_image = clean_image.view(-1, *clean_image.shape[2:])

        return clean_image, top_loss




    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 1.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        # return_dict: bool = True,
        num_query_tokens = 16, # [FIXED] Should be changed according to the model
        guidance = False, 
        target_class = None,
        update_seq = False,
        max_seq_len = 150,
        verbose=False,
        recurrence_step = 1,
        guidance_start_step = 0,
        guidance_stop_step = None,
        **decoder_sample_kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        assert eta > 0, "eta should be greater than 0"
        

        # Sample gaussian noise to begin loop
        if "sample_size" in self.unet.config: 
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size * self.active_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (batch_size * self.active_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        
        else: # [TransformerDenoiser]
            image_shape = (
                batch_size * self.active_size,
                num_query_tokens,
                self.unet.config.in_channels
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if guidance_stop_step is None:
            guidance_stop_step = num_inference_steps
        else:
            guidance_stop_step = min(guidance_stop_step, num_inference_steps)

        progress_bar = self.progress_bar(self.scheduler.timesteps)
        for denoise_step, t in enumerate(progress_bar):
            # TODO: change the device of t to cuda
            # 1.0 add guidance
            if guidance and self.classifier_model is not None and denoise_step >= guidance_start_step and denoise_step < guidance_stop_step:
                # Reference from Universal Guidance
                # get gradient
                for i in range(recurrence_step):
                    if i == 0:
                        image_t = image # z_t (image)
                    else: 
                        image_t = self.scheduler.forward_step(t, image) # build from z_t-1 (image)

                    if self.guidance_type == "seq":
                        gradient, model_output = self.get_gradient(t, image_t, target, update_seq=update_seq, max_seq_len=max_seq_len, verbose=verbose, **decoder_sample_kwargs)
                    else: # "latent"
                        gradient, model_output = self.get_gradient_latent(t, image_t, target_class, verbose=verbose) ## model_output detach() ？
                    # add gradient to model_output

                    noise_scale = self.scheduler.get_random_noise_scale(t, eta)

                    # model_output = model_output + self.guidance_scale * self.scheduler.get_noise_scale(t) *  gradient

                    # use normalization technique
                    eps = 1e-8
                    grad_norm = torch.linalg.norm(gradient.view(gradient.size(0), -1), dim=1)
                    _, l, d = gradient.size()
                    r =  noise_scale * torch.sqrt(torch.tensor(l * d))
                    gradient = r * gradient / (grad_norm.view(-1, 1, 1) + eps) 

                    model_output = model_output +  self.guidance_scale * gradient

                    # print("grad norm:", grad_norm.mean().item())
                    # print(torch.sqrt(torch.tensor(l * d)).item())


                    ## tree search process
                    ## branch out

                    ## TODO: check t == 0

                    image = self.scheduler.step(
                        model_output, t, image_t, branch_size=self.branch_size, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                    ).prev_sample  # z_t-1  shape: [batch_size * active_size * branch_size, *image_shape]

                    # print("image shape: ", image.shape)
            
                    ## select the top acitve_size
                    

                    # if self.branch_size > 1:
                    if True:
                        if denoise_step + 1 < len(self.scheduler.timesteps):
                            prev_t = self.scheduler.timesteps[denoise_step + 1]
                            image, loss = self.evaluate_then_select(batch_size, prev_t, image, target_class)
                        else: 
                            image, loss = self.select_clean(batch_size, self.active_size, image, target_class)

                        progress_bar.set_postfix({'loss': f'{loss.mean():.3f}'}, refresh=False)


                        
                        



            else:
                # 1. predict noise model_output
                model_output = self.unet(image, t).sample # noise prediction
                # print("model output norm: ", torch.norm(model_output))
                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t, image, branch_size=self.branch_size, eta=eta, 
                    use_clipped_model_output=use_clipped_model_output, generator=generator
                ).prev_sample

                

  

                if self.branch_size > 1:
                    if denoise_step + 1 < len(self.scheduler.timesteps):
                        prev_t = self.scheduler.timesteps[denoise_step + 1]
                        image, _  = self.evaluate_then_select(batch_size, prev_t, image, target_class)
                    else: 
                        image, _ = self.select_clean(batch_size, self.active_size, image, target_class)
                    


        # postprocessing here # TODO

        image, loss = self.select_clean(batch_size, 1, image, target_class)

        print("final loss: ", loss.mean().item())


        return image
    






class GuidedSearchDDIMPipeline1D(GuidedDDIMPipeline1D):
    def __init__(
        self,
        unet,
        scheduler,
        classifier_model_config=None,
        encdec=None,
        loss_type="l2",
        guidance_scale=1.0,
        active_size=1,
        branch_size=1,
        value_func=None,
    ):
       
        # Store the additional parameters for the child class.
        self.active_size = active_size
        self.branch_size = branch_size
        self.value_func = value_func


        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMSchedulerCustomMutiple.from_config(scheduler.config)


        self.register_modules(unet=unet, scheduler=scheduler)
        self.classifier_model = None
        if classifier_model_config is not None:
            self.classifier_model = instantiate_from_config(classifier_model_config)
            self.classifier_type = self.classifier_model.classifier_type
            self.guidance_type = self.classifier_model.guidance_type
        self.encdec = encdec
        self.softmax = torch.nn.Softmax(dim=-1)
        self.guidance_scale = guidance_scale
        if loss_type == "l2":
            self.loss = nn.MSELoss()
            self.loss_func = nn.MSELoss(reduction='none')
        else:
            # self.loss = neg_reward_loss()
            self.loss = nn.CrossEntropyLoss()
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        if self.encdec is not None:
            self.decoder_a_id = self.encdec.dec_tokenizer.token_to_id("A") # 5 


    def evaluate_then_select(self, batch_size, t, image, max_seq_len=150, **decoder_sample_kwargs):

        model_output = self.unet(image, t).sample
        original_sample = self.scheduler.get_original_sample(model_output, t, image)

        _, seqs = self.encdec.generate(
                original_sample, max_seq_len,
                **decoder_sample_kwargs
        )

        values = self.value_func(seqs)
        # tran value:list to tensor
        values = torch.tensor(values, device=image.device)

        loss = -values

        ## select the top active_size
        loss = loss.view(batch_size, -1)
        topk_loss, topk_idx = torch.topk(loss, self.active_size, dim=1, largest=False)

        image = image.view(batch_size, -1, *image.shape[1:])
        image = image[torch.arange(batch_size).unsqueeze(1), topk_idx]
        image = image.view(-1, *image.shape[2:])

        return image, topk_loss
    

    def select_clean(self, batch_size, top_size, clean_image, max_seq_len=150, **decoder_sample_kwargs):
        _, seqs = self.encdec.generate(
                clean_image, max_seq_len,
                **decoder_sample_kwargs
        )

        values = self.value_func(seqs)
        values = torch.tensor(values, device=clean_image.device)
        loss = -values

        ## select the top active_size
        loss = loss.view(batch_size, -1)
        top_loss, top_idx = torch.topk(loss, top_size, dim=1, largest=False)

        clean_image = clean_image.view(batch_size, -1, *clean_image.shape[1:])
        clean_image = clean_image[torch.arange(batch_size).unsqueeze(1), top_idx]
        clean_image = clean_image.view(-1, *clean_image.shape[2:])

        return clean_image, top_loss




    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 1.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        # return_dict: bool = True,
        num_query_tokens = 16, # [FIXED] Should be changed according to the model
        guidance = False, 
        target_class = None,
        update_seq = False,
        max_seq_len = 150,
        verbose=False,
        recurrence_step = 1,
        guidance_start_step = 0,
        guidance_stop_step = None,
        normalize_gradient = False,
        **decoder_sample_kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        assert eta > 0, "eta should be greater than 0"
        

        # Sample gaussian noise to begin loop
        if "sample_size" in self.unet.config: 
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size * self.active_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (batch_size * self.active_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        
        else: # [TransformerDenoiser]
            image_shape = (
                batch_size * self.active_size,
                num_query_tokens,
                self.unet.config.in_channels
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if guidance_stop_step is None:
            guidance_stop_step = num_inference_steps
        else:
            guidance_stop_step = min(guidance_stop_step, num_inference_steps)

        progress_bar = self.progress_bar(self.scheduler.timesteps)
        for denoise_step, t in enumerate(progress_bar):
            # TODO: change the device of t to cuda
            # 1.0 add guidance
            if guidance and self.classifier_model is not None and denoise_step >= guidance_start_step and denoise_step < guidance_stop_step:
                # Reference from Universal Guidance
                # get gradient
                for i in range(recurrence_step):
                    if i == 0:
                        image_t = image # z_t (image)
                    else: 
                        image_t = self.scheduler.forward_step(t, image) # build from z_t-1 (image)

                    if self.guidance_type == "seq":
                        gradient, model_output = self.get_gradient(t, image_t, target, update_seq=update_seq, max_seq_len=max_seq_len, verbose=verbose, **decoder_sample_kwargs)
                    else: # "latent"
                        gradient, model_output = self.get_gradient_latent(t, image_t, target_class, verbose=verbose) ## model_output detach() ？
                    # add gradient to model_output


                    if normalize_gradient:
                        # use normalization technique
                        eps = 1e-8
                        noise_scale = self.scheduler.get_random_noise_scale(t, eta)
                    
                        grad_norm = torch.linalg.norm(gradient.view(gradient.size(0), -1), dim=1)
                        _, l, d = gradient.size()
                        r =  noise_scale * torch.sqrt(torch.tensor(l * d))
                        gradient = r * gradient / (grad_norm.view(-1, 1, 1) + eps) 

                        model_output = model_output +  self.guidance_scale * gradient

                    else: 
                        model_output = model_output + self.guidance_scale * self.scheduler.get_noise_scale(t) *  gradient


                    # print("grad norm:", grad_norm.mean().item())
                    # print(torch.sqrt(torch.tensor(l * d)).item())
            else:
                # 1. predict noise model_output
                model_output = self.unet(image, t).sample # noise prediction
                # print("model output norm: ", torch.norm(model_output))
                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # image = self.scheduler.step(
                #     model_output, t, image, branch_size=self.branch_size, eta=eta, 
                #     use_clipped_model_output=use_clipped_model_output, generator=generator
                # ).prev_sample
                image_t = image

            ## tree search process
            ## branch out

            image = self.scheduler.step(
                model_output, t, image_t, branch_size=self.branch_size, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample  # z_t-1  shape: [batch_size * active_size * branch_size, *image_shape]

            
    
            ## select the top acitve_size
            if self.branch_size > 1:
            # if True:
                if denoise_step + 1 < len(self.scheduler.timesteps):
                    prev_t = self.scheduler.timesteps[denoise_step + 1]
                    image, loss = self.evaluate_then_select(batch_size, prev_t, image, max_seq_len, **decoder_sample_kwargs)
                else: 
                    image, loss = self.select_clean(batch_size, self.active_size, image, max_seq_len, **decoder_sample_kwargs)

                progress_bar.set_postfix({'loss': f'{loss.mean():.3f}'}, refresh=False)

        # postprocessing here # TODO

        image, loss = self.select_clean(batch_size, 1, image, max_seq_len, **decoder_sample_kwargs)

        print("final loss: ", loss.mean().item())

        return image
    
