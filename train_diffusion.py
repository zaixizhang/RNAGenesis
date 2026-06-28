# =============================================================================
# RNAGenesis - latent diffusion (pre-)training and fine-tuning script.
#
# This trains / fine-tunes the latent diffusion model (the denoiser) that runs
# on top of the frozen RNAGenesis sequence auto-encoder (`EncDec`). The exact
# same model objects used by `generation.py` for inference (`EncDec`,
# `--enc_dec_file` / `--dm_file`, the DDIM scheduler and `DDIMPipeline1D`) are
# reused here, so a checkpoint produced by this script can be fed directly to
# `generation.py` via `--dm_file`.
#
# Two modes:
#   * train from scratch  -> pass `--model_config_name_or_path` (e.g.
#                            configs/diffusion/config_n.json)
#   * fine-tune           -> pass `--pretrained_ckpts <dm_file>` to continue
#                            training a released diffusion checkpoint on new data
#                            (typically with `--lr_warmup_steps` and a few epochs).
#
# See FINETUNE.md for a step-by-step walkthrough.
#
# modified from https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py, git commit e9eb093
# Initialization Codes
import os 
from util import XDG_CACHE_HOME
os.environ['XDG_CACHE_HOME'] = XDG_CACHE_HOME # (Optional) to specify where to store the huggingface model/ dataset caches. 
import sys 
sys.path.append("models/autoencoder/encoder") # for the dependencies of UTR-LM

import argparse
import logging
import math
import shutil
from datetime import timedelta
import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm

import diffusers
from models.diffusion_models.pipeline_ddim import DDIMPipeline1D
from diffusers import DDIMScheduler 
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from data.dataset_builder import build_dataset, build_dataset_rna, build_reward_dataset
from models.autoencoder.encdec import EncDec, DataCollatorEncDec
import time
from models.diffusion_models.util import get_model

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument( "--model_config_name_or_path", type=str, help="The config of the UNet model to train",)
    parser.add_argument( "--encdec_checkpoint", type=str, default="./exp-sb", help="The checkpoint of the pretrained encoder-decoder model",)
    parser.add_argument("--scheduler_config", type=str, default='models/diffusion_models/config_ddim.json')
    parser.add_argument("--init_noise", type=float, default=0.0)

    # parser.add_argument("--train_data", type=str, default=None, help=( "training data file"),)
    parser.add_argument("--train_data", nargs='+', type=str, default=None, help=( "training data file"),)
    parser.add_argument( "--output", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--overwrite_output_dir", action="store_true")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--save_model_epochs", type=int, default=1, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    ) 

    parser.add_argument(
        "--data_type",
        type=str,
        default="protein", 
        help="The type of data to train on. Choose between ['protein', 'rna']",
    )
    parser.add_argument(
        "--save_all_epochs",
        action="store_true",
        help="Whether to save the model at the end of every epoch.",
    )
    parser.add_argument(   #[YYK] add the pretrained checkpoints
        "--pretrained_ckpts",
        type=str,
        default=None,
        help="The path to the pretrained checkpoints.",
    )
    parser.add_argument(
        "--remove_pad",
        action="store_true",
        help="Whether to remove the <pad> token in the original data.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="The type of dataset to use. Choose between ['protein', 'rna', 'utr_reward']",
    )
    parser.add_argument(
        "--reward_name",
        type=str,
        default="te_log", # "rl_log2"
        help="The name of the reward in the dataset.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="TransformerDenoiser",
        help="The class of the model to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set in the training script.",
    )
    parser.add_argument(
        "--full_deterministic",
        action="store_true",
        help="Whether to use full deterministic algorithms.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args




def main(args):
    # [0627] set the random seed and use deterministic algorithms (optional) for reproducibility
    accelerate.utils.set_seed(args.seed)
    if args.full_deterministic: # Only for debugging, no need for single GPU training
        ###
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ###
        torch.use_deterministic_algorithms(True)

    logging_dir = os.path.join(args.output, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, # TODO: check if the mixed_precision is needed
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        # wandb.init(project="rnadiff")
        # name = f"{args.model_name_or_path.split('/')[-1]}-{args.data_type}"

        # wandb.init(settings=wandb.Settings(start_method="fork"))
        # wandb.login(
        #     key = "87d66c7d19e7fb4359815928318adca7ef35c666"
        #     )

    model_class = get_model(args.model_class)
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                if not isinstance(model , EncDec): # skip frozen EncDec model
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), model_class)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if not isinstance(model, EncDec):
                # load diffusers style into model
                    load_model = model_class.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)

    
    # Initialize Encoder-Decoder
    encdec_model = EncDec.from_pretrained(args.encdec_checkpoint)
    assert encdec_model.data_type == args.data_type

    collator = DataCollatorEncDec(
        encdec_model.qt.config.num_query_tokens if encdec_model.qt is not None else 1, ## [YYK] update 0802
        data_type=args.data_type,
        rna_encoder_type=encdec_model.rna_encoder_type,
        rna_tokenizer_config=encdec_model.rna_tokenizer_config)

    # Initialize the model
    print(f"Load {args.model_class} model")
    if args.pretrained_ckpts is None:
        if args.model_config_name_or_path is None: # [YYK] previous version
            #raise ValueError("model config file is needed!")
            model = model_class(in_channels = encdec_model.qt.config.hidden_size)
        else:
            config = model_class.load_config(args.model_config_name_or_path)
            config["in_channels"] = encdec_model.qt.config.hidden_size # [YYk] set the in_channels
            model = model_class.from_config(config)
    else:
        model = model_class.from_pretrained(args.pretrained_ckpts, subfolder="unet")

    assert model.config.in_channels == encdec_model.qt.config.hidden_size
    if args.model_class == "DiT":
        assert model.config.image_size == encdec_model.qt.config.num_query_tokens

    if args.logger == "wandb":
        time_ = time.strftime("%m-%d-%H:%M:%S", time.localtime())
        wandb.init(
            project="biodiff", 
            # name = f"dm-{encdec_model.qt.config.hidden_size}-hidden-size-{args.data_type}-data-type-{time_}"
            )
    
    # Create EMA for the model.
    if args.use_ema:
        if args.pretrained_ckpts is not None and os.path.exists(os.path.join(args.pretrained_ckpts, "unet_ema")):
            ema_model = EMAModel.from_pretrained(os.path.join(args.pretrained_ckpts, "unet_ema"), model_class)
        else:
            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=model_class,
                model_config=model.config,
            )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler [?] DDPMScheduler
    noise_scheduler_config = DDIMScheduler.load_config(args.scheduler_config)
    noise_scheduler = DDIMScheduler.from_config(noise_scheduler_config)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    # dataset = build_dataset(args.train_data) #TODO add my tensor dataset.
    if args.dataset_type is None:
        args.dataset_type = args.data_type
    if args.dataset_type == "protein": ##[DEPRECATED]
        dataset = build_dataset(args.train_data) 
    elif args.dataset_type == "rna":  
        dataset = build_dataset_rna(
            args.train_data, 
            encoder_type=encdec_model.rna_encoder_type,
            tokenizer_config=encdec_model.rna_tokenizer_config) 
    elif args.dataset_type == "utr_reward":
        dataset=build_reward_dataset(
            args.train_data, 
            encoder_type=encdec_model.rna_encoder_type,
            tokenizer_config=encdec_model.rna_tokenizer_config, split=True,
            reward_name=args.reward_name,
            remove_pad=args.remove_pad)

    logger.info(f"Dataset size: {len(dataset['train'])}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, collate_fn = collator
    )

    # [MOVED]
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        ## [FIXED]
        num_warmup_steps=args.lr_warmup_steps, # * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, encdec_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, encdec_model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=args)

    # [MOVE UPWARDS]
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch 
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with torch.no_grad():
                clean_images = encdec_model.get_latent(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
                init_noise = torch.randn(
                    clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
                ).to(clean_images.device)
                clean_images = clean_images + init_noise * args.init_noise
            noise = torch.randn(
                clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
            ).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                elif noise_scheduler.config.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    velocity = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                    loss = F.mse_loss(model_output, velocity) # this could have different weights!
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1) #update only once per gradient accumulation
                global_step += 1

                if global_step % args.checkpointing_steps == 0: # For resuming training
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None: # 1 would be fine
                        checkpoints = os.listdir(args.output)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # save the model
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDIMPipeline1D(
                    unet=unet,
                    scheduler=noise_scheduler,
                )
                if not args.save_all_epochs:
                    save_ckpt_path = args.output
                else:
                    save_ckpt_path = os.path.join(args.output, f"epoch-{epoch}") #TODO: check if this is correct
                os.makedirs(save_ckpt_path, exist_ok=True)
                pipeline.save_pretrained(save_ckpt_path)

                if args.use_ema:
                    ema_model.restore(unet.parameters())


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
