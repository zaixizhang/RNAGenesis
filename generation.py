# Initialization Codes
import os 
from util import XDG_CACHE_HOME
os.environ['XDG_CACHE_HOME'] = XDG_CACHE_HOME # (Optional) specify where to store the huggingface model/ dataset caches. 
import sys 
sys.path.append("models/autoencoder/encoder") # for the dependencies of UTR-LM; if we will not use UTR-LM, we can remove this line

import numpy as np
import torch
import argparse
from models.autoencoder.encdec import EncDec
from models.diffusion_models.pipeline_ddim import GuidedDDIMPipeline1D, GuidedSearchDDIMPipeline1D, SearchDDIMPipeline1D
from omegaconf import OmegaConf
from util import set_random_seed, count_params

from test_wrap_aptamer import similarity_value_function, mfe_value_function, combine_value_function

def inference(args, unknown, device="cuda"):
    if args.batch_num <= 0: 
        print("batch_num should be greater than 0")
        return
    
    # 1. Load Models
    ## Load the encoder-decoder model
    if args.decoder_config is not None:
        encdec = EncDec.from_pretrained(args.enc_dec_file, decoder_config=args.decoder_config)
    else:
        encdec = EncDec.from_pretrained(args.enc_dec_file)
    encdec.to(device)
    encdec.eval()
    print("EncDec Model:")
    count_params(encdec, print_modules=True)

    if args.guidance and args.guidance_classifier_model_config is not None:
        ### Load the config for the guidance classifier model
        config = OmegaConf.load(args.guidance_classifier_model_config)
        unknown_omegaconf = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(config, unknown_omegaconf)
    else:
        config = None

    ## Load the diffusion model
    if args.search_general:
        assert args.search_goal in ["similarity", "mfe", "combine"]
        if args.search_goal == "similarity":
            value_function = similarity_value_function
        elif args.search_goal == "mfe":
            value_function = mfe_value_function
        elif args.search_goal == "combine":
            value_function = combine_value_function
        print(f"search_goal: {args.search_goal}")
        print(f"active_size: {args.active_size}, branch_size: {args.branch_size}")

        diffusion_pipeline = GuidedSearchDDIMPipeline1D.from_pretrained(
            args.dm_file, 
            classifier_model_config = config, ## guidance classifier model config
            loss_type = args.classifier_loss_type, ## classifier loss type
            encdec=encdec, ## only useful for sequence-level guidance
            guidance_scale = args.guidance_scale, 
            active_size = args.active_size,
            branch_size = args.branch_size,
            value_func=value_function,
            )
        
    elif args.tree_search:
        diffusion_pipeline = SearchDDIMPipeline1D.from_pretrained(
            args.dm_file, 
            classifier_model_config = config, ## guidance classifier model config
            loss_type = args.classifier_loss_type, ## classifier loss type
            encdec=encdec, ## only useful for sequence-level guidance
            guidance_scale = args.guidance_scale, 
            active_size = args.active_size,
            branch_size = args.branch_size,
            )
    else:
        diffusion_pipeline = GuidedDDIMPipeline1D.from_pretrained(
            args.dm_file, 
            classifier_model_config = config, ## guidance classifier model config
            loss_type = args.classifier_loss_type, ## classifier loss type
            encdec=encdec, ## only useful for sequence-level guidance
            guidance_scale = args.guidance_scale, 
            )
        
    print("Diffusion Model:")
    model = diffusion_pipeline.unet 
    count_params(model)
    assert model.config.in_channels == encdec.config.hidden_size
    model.to(device)
    model.eval()

    ## configurate the guidance classifier model
    guidance_classifier_model = diffusion_pipeline.classifier_model
    guidance_remark = "_wo_guidance"
    if guidance_classifier_model is not None:
        try:
            guidance_classifier_model.to(device)
            guidance_classifier_model.eval()
        except Exception as e:
            print(e)
            guidance_classifier_model.model.to(device)
            guidance_classifier_model.model.eval()
        guidance_classifier_model.device = device
        guidance_remark = f"_{guidance_classifier_model.classifier_type}_guidance_target_class_{guidance_classifier_model.guidance_type}_{os.path.splitext(os.path.basename(args.guidance_classifier_model_config))[0]}_{args.target_class}_{args.guidance_scale}"
        guidance_remark += f"_{args.recurrence_step}_{args.guidance_start_step}_{args.guidance_stop_step}"

    generator = torch.Generator(device=diffusion_pipeline.device).manual_seed(args.seed)

    ####
    decoder_sample_kwargs = {
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "eos_token": args.eos_token,
        "bos_token": args.bos_token,
        "truncate": args.truncate,
        "extend_input": args.extend_input,
        "multiple_eos_tokens": args.multiple_eos_tokens,
    }
    latent_vecs_list = []
    with torch.no_grad():
        for i in range(args.batch_num):
            # run pipeline in inference (sample random noise and denoise)
            latent_gen = diffusion_pipeline( 
                batch_size = args.batch_size,
                generator=generator,
                eta = args.eta,
                num_inference_steps = args.num_inference_steps,
                num_query_tokens = encdec.config.num_query_tokens,
                guidance = args.guidance,
                target_class = args.target_class,
                ### guidance related
                update_seq = args.update_seq,
                verbose = False,
                max_seq_len = args.max_seq_len,
                recurrence_step = args.recurrence_step,
                guidance_start_step = args.guidance_start_step,
                guidance_stop_step = args.guidance_stop_step,
                **decoder_sample_kwargs, ##TODO: check if it should be same as the real sampling process
            )
            latent_vecs_list.append(latent_gen)
            # 2. decode the latent space to the original space

            # print(f"batch {i} latent_gen shape: {latent_gen.shape}")

            logits_, seqs_ = encdec.generate(
                latent_gen, args.max_seq_len, 
                **decoder_sample_kwargs,
                )

            if i == 0:
                seqs = seqs_
            else:
                seqs.extend(seqs_)

            # print(f"batch {i} seqs shape: {len(seqs)}")

            # ## print the type of seqs
            # print(f"seqs type: {type(seqs)}")
            # print(f"seqs[0] type: {type(seqs[0])}")
        
        save_folder = os.path.join(args.superfolder, args.mid_folder)
        os.makedirs(save_folder, exist_ok=True)  

        latent_vecs = torch.cat(latent_vecs_list, dim=0)
        if args.save_latent_space:
            latent_vecs_ = latent_vecs.cpu().numpy()
            np.save(os.path.join(save_folder, f"gen_latent_vecs_{len(seqs)}.npy"), latent_vecs_)
        
        # Save the generated Samples
        if not args.do_sample:
            sample_strategy = "argmax_(greedy)"
        else:
            if args.top_p < 1.0:
                sample_strategy = f"top_p={args.top_p}"
            elif args.top_k > 0:
                sample_strategy = f"top_k={args.top_k}"
        print("sample strategy", sample_strategy)
        if args.guidance:
            print("with guidance: guidance scale {}".format(args.guidance_scale))
        else:
            print("without guidance")

        if args.search_general:
            # sample_remark = f"_active_{args.active_size}_branch_{args.branch_size}"

            print(f"tree search: active_size {args.active_size}, branch_size {args.branch_size}")
        else:
            print("no tree search")

        sample_remark = f"_{sample_strategy}_{args.eos_token}_{len(seqs)}"
        print(f"Saving generated sequences to {save_folder}")

        if args.search_general:
            save_path = os.path.join(
                save_folder,
                f"seqs{guidance_remark}{sample_remark}{args.remark}_active_{args.active_size}_branch_{args.branch_size}.txt")
        else:
            save_path = os.path.join(
                save_folder,
                f"seqs{guidance_remark}{sample_remark}{args.remark}.txt")
        

        with open(save_path, "w") as f:
            for seq in seqs:
                f.write(seq + "\n")
        print("DONE")


def parse_args():
    parser = argparse.ArgumentParser()
    ### important arguments
    parser.add_argument('--enc_dec_file', type=str, required=True)
    parser.add_argument('--dm_file', type=str, required=True)
    parser.add_argument('--num_inference_steps', type=int, required=True, help="number of DDIM steps")
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--batch_num', type=int, required=True, help="batch_num should be greater than 0")

    parser.add_argument('--seed', type=int, required=True)

    # ddim related
    parser.add_argument('--eta', type=float, required=True)

    ## guidance related
    parser.add_argument('--guidance', action="store_true")
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--guidance_classifier_model_config', type=str, required=True)
    parser.add_argument('--guidance_scale', type=float, required=True)
    parser.add_argument('--classifier_loss_type', type=str, required=True)

    parser.add_argument('--recurrence_step', type=int, required=True)
    parser.add_argument('--guidance_start_step', type=int, required=True)
    parser.add_argument('--guidance_stop_step', type=int, required=True)
    parser.add_argument('--update_seq', action="store_true")


    ## tree search related
    # value-ts
    parser.add_argument('--search_general', action="store_true")
    parser.add_argument('--search_goal', type=str, required=True)

    # ts
    parser.add_argument('--tree_search', action="store_true")
    parser.add_argument('--active_size', type=int, required=True)
    parser.add_argument('--branch_size', type=int, required=True)

    # sampling related
    parser.add_argument('--max_seq_len', type=int, required=True)
    parser.add_argument('--do_sample', action="store_true")
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--top_k', type=int, required=True)   
    parser.add_argument('--eos_token', type=str, required=True)
    parser.add_argument('--bos_token', type=str, required=True)
    parser.add_argument('--multiple_eos_tokens', action="store_true")
    parser.add_argument('--truncate', action="store_true")
    parser.add_argument('--extend_input', action="store_true")

    parser.add_argument('--remark', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--mid_folder', type=str, required=True)
    parser.add_argument('--superfolder', type=str, required=True)
    
    parser.add_argument("--decoder_config", type=str, required=True)
    # store latent space
    parser.add_argument('--save_latent_space', action="store_true")

    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()
    device= args.device if torch.cuda.is_available() else 'cpu'
    set_random_seed(args.seed)
    inference(args, unknown, device=device)
