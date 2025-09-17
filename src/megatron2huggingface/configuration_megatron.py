"""Megatron model configuration - generated from megatron2huggingface"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class MegatronConfig(PretrainedConfig):
    r"""This configures a MegatronModel.

    Args:
        num_layers (`int`):
            Argument num_layers. Number of transformer layers.
        encoder_num_layers (`int`):
            Argument encoder_num_layers. Number of encoder transformer layers.
        decoder_num_layers (`int`):
            Argument decoder_num_layers. Number of decoder transformer layers.
        hidden_size (`int`):
            Argument hidden_size. Tansformer hidden size.
        ffn_hidden_size (`int`):
            Argument ffn_hidden_size. Transformer Feed-Forward Network hidden size. This is set to 4*hidden-size if
            not provided
        num_attention_heads (`int`):
            Argument num_attention_heads. Number of transformer attention heads.
        attention_backend (`str`, *optional*, defaults to "default"):
            Argument attention_backend. Attention backend to use (flash,fused,unfused,local,auto). Defaults to auto
        kv_channels (`int`):
            Argument kv_channels. Projection weights dimension in multi-head attention. This is set to
             args.hidden_size // args.num_attention_heads if not provided.
        group_query_attention (`bool`, *optional*, defaults to `False`):
            Argument group_query_attention. Use group-query attention.
        num_query_groups (`int`, *optional*, defaults to 1):
            Argument num_query_groups.
        max_position_embeddings (`int`):
            Argument max_position_embeddings. Maximum number of position embeddings to use. This is the size
            of position embedding.
        position_embedding_type (`str`, *optional*, defaults to "learned_absolute"):
            Argument position_embedding_type. Position embedding type.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Argument relative_attention_num_buckets. Number of buckets for relative position embeddings.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            Argument relative_attention_max_distance. Maximum distance for relative position embeddings calculation.
        use_rotary_position_embeddings (`bool`, *optional*, defaults to `False`):
            Argument use_rotary_position_embeddings. Use rotary positional embeddings or not. Deprecated:
            use --position-embedding-type
        rotary_base (`int`, *optional*, defaults to 10000):
            Argument rotary_base. Base to use for rotary positional embeddings, default 10000
        rotary_percent (`float`, *optional*, defaults to 1.0):
            Argument rotary_percent. Percent of rotary dimension to use, default 100%%
        rotary_interleaved (`bool`, *optional*, defaults to `False`):
            Argument rotary_interleaved. Use interleaved rotary embedding.
        rotary_seq_len_interpolation_factor (`int`):
            Argument rotary_seq_len_interpolation_factor. Sequence length interpolation factor for rotary embeddings.
        use_rope_scaling (`bool`, *optional*, defaults to `False`):
            Argument use_rope_scaling. Apply rope scaling as used in llama3.x
        rope_scaling_factor (`float`, *optional*, defaults to 8.0):
            Argument rope_scaling_factor. Rope scaling factor in llama3.x models
        no_rope_freq (`Any`):
            Argument no_rope_freq. Controls which layers to skip performing Rotary Position Embedding. Accepts
            either: - An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers. - A
            string containing a Python list expression that defines a custom pattern, e.g.: "([0]*3+[1]*1)*3"
            evaluates to [0,0,0,1,0,0,0,1,0,0,0,1] where 1 indicates no-rope layer. This patten is equivalent
            to --no-rope-freq=4.By default this is disabled and set to None, indicating RoPE will be performedon
            every layer.
        add_position_embedding (`bool`, *optional*, defaults to `True`):
            Argument add_position_embedding. Disable position embedding. Deprecated: use --position-embedding-type
        mrope_section (`list[int]`):
            Argument mrope_section. Multimodal rope section is for channel dimension, empty by default.
        make_vocab_size_divisible_by (`int`, *optional*, defaults to 128):
            Argument make_vocab_size_divisible_by. Pad the vocab size to be divisible by this value.This is added
            for computational efficieny reasons.
        normalization (`str`, *optional*, defaults to "LayerNorm"):
            Argument normalization. Which normalization technique to use.
        norm_epsilon (`float`, *optional*, defaults to 1e-05):
            Argument norm_epsilon. Epsilon for layer norm and RMS norm.
        apply_layernorm_1p (`bool`, *optional*, defaults to `False`):
            Argument apply_layernorm_1p. Adjust LayerNorm weights such that they are centered around zero.
            This improves numerical stability.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            Argument apply_residual_connection_post_layernorm. If set, use original BERT residula connection ordering.
        openai_gelu (`bool`, *optional*, defaults to `False`):
            Argument openai_gelu. Use OpenAIs GeLU implementation. This optionshould not be used unless for
            backward compatibilityreasons.
        squared_relu (`bool`, *optional*, defaults to `False`):
            Argument squared_relu. Use squared relu activation instead of default gelu
        swiglu (`bool`, *optional*, defaults to `False`):
            Argument swiglu. Use gated linear units and SiLU activation instead of default gelu
        onnx_safe (`bool`):
            Argument onnx_safe. Use workarounds for known problems with Torch ONNX exporter
        bert_binary_head (`bool`, *optional*, defaults to `True`):
            Argument bert_binary_head. Disable BERT binary head.
        untie_embeddings_and_output_weights (`bool`, *optional*, defaults to `False`):
            Argument untie_embeddings_and_output_weights. Untie embeddings and output weights.
        multi_latent_attention (`bool`, *optional*, defaults to `False`):
            Argument multi_latent_attention. Use multi-latent attention for model.
        mtp_num_layers (`int`):
            Argument mtp_num_layers. Number of Multi-Token Prediction (MTP) Layers.MTP extends the prediction scope
            to multiple future tokens at each position.This MTP implementation sequentially predict additional
            tokens by using D sequential modules to predict D additional tokens.
        mtp_loss_scaling_factor (`float`, *optional*, defaults to 0.1):
            Argument mtp_loss_scaling_factor. Scaling factor of Multi-Token Prediction (MTP) loss. We compute
            the average of the MTP losses across all depths, and multiply it the scaling factor to obtain the
            overall MTP loss, which serves as an additional training objective.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Argument attention_dropout. Post attention dropout probability.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Argument hidden_dropout. Dropout probability for hidden state transformer.
        weight_decay (`float`, *optional*, defaults to 0.01):
            Argument weight_decay. Weight decay coefficient for L2 regularization.
        start_weight_decay (`float`):
            Argument start_weight_decay. Initial weight decay coefficient for L2 regularization.
        end_weight_decay (`float`):
            Argument end_weight_decay. End of run weight decay coefficient for L2 regularization.
        weight_decay_incr_style (`str`, *optional*, defaults to "constant"):
            Argument weight_decay_incr_style. Weight decay increment function.
        clip_grad (`float`, *optional*, defaults to 1.0):
            Argument clip_grad. Gradient clipping based on global L2 norm.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            Argument adam_beta1. First coefficient for computing running averages of gradient and its square
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            Argument adam_beta2. Second coefficient for computing running averages of gradient and its square
        adam_eps (`float`, *optional*, defaults to 1e-08):
            Argument adam_eps. Term added to the denominator to improvenumerical stability
        sgd_momentum (`float`, *optional*, defaults to 0.9):
            Argument sgd_momentum. Momentum factor for sgd
        micro_batch_size (`int`):
            Argument micro_batch_size. Batch size per model instance (local batch size). Global batch size is
            local batch size times data parallel size times number of micro batches.
        batch_size (`int`):
            Argument batch_size. Old batch size parameter, do not use. Use --micro-batch-size instead
        global_batch_size (`int`):
            Argument global_batch_size. Training batch size. If set, it should be a multiple of micro-batch-size
            times data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size as
            the global batch size. This choice will result in 1 for number of micro-batches.
        rampup_batch_size (`list[str]`):
            Argument rampup_batch_size. Batch size ramp up with the following values:  --rampup-batch-size <start
            batch size>                       <batch size incerement>                       <ramp-up samples>
            For example:   --rampup-batch-size 16 8 300000 \    --global-batch-size 1024will start with global
            batch size 16 and over  (1024 - 16) / 8 = 126 intervals will increasethe batch size linearly to 1024.
            In each intervalwe will use approximately 300000 / 126 = 2380 samples.
        decrease_batch_size_if_needed (`bool`, *optional*, defaults to `False`):
            Argument decrease_batch_size_if_needed. If set, decrease batch size if microbatch_size * dp_sizedoes
            not divide batch_size. Useful for KSO (Keep Soldiering On)to continue making progress if number of
            healthy GPUs (andcorresponding dp_size) does not support current batch_size.Old batch_size will be
            restored if training is re-started withdp_size that divides batch_size // microbatch_size.
        recompute_activations (`bool`, *optional*, defaults to `False`):
            Argument recompute_activations. recompute activation to allow for training with larger models,
            sequences, and batch sizes.
        recompute_granularity (`str`):
            Argument recompute_granularity. Checkpoint activations to allow for training with larger models,
            sequences, and batch sizes. It is supported at two granularities 1) full: whole transformer layer
            is recomputed, 2) selective: submodules set in --recompute-modules are recomputed, default is core_attn.
        check_for_nan_in_loss_and_grad (`bool`, *optional*, defaults to `True`):
            Argument check_for_nan_in_loss_and_grad. Check for NaNs in loss and grad
        check_for_spiky_loss (`bool`, *optional*, defaults to `False`):
            Argument check_for_spiky_loss. Check for spiky loss
        check_for_large_grads (`bool`, *optional*, defaults to `False`):
            Argument check_for_large_grads. Check for unexpectedly large grads
        distribute_saved_activations (`bool`, *optional*, defaults to `False`):
            Argument distribute_saved_activations. If set, distribute recomputed activations across model
            parallel group.
        recompute_method (`str`):
            Argument recompute_method. 1) uniform: uniformly divide the total number of Transformer layers
            and recompute the input activation of each divided chunk at specified granularity, 2) recompute the
            input activations of only a set number of individual Transformer layers per pipeline stage and do the
            rest without any recomputing at specified granularitydefault) do not apply activations recompute to
            any layers
        recompute_num_layers (`int`):
            Argument recompute_num_layers. 1) uniform: the number of Transformer layers in each uniformly
            divided recompute unit, 2) block: the number of individual Transformer layers to recompute within
            each pipeline stage.
        recompute_modules (`list[str]`):
            Argument recompute_modules. The submodules to recompute. choices: "core_attn", "moe_act",
            "layernorm", "mla_up_proj", "mlp", "moe". default: ["core_attn"]."core_attn": recompute the core
            attention part of the transformer layer. "moe_act": recompute the MoE MLP activation function.
            "layernorm": recompute the input_layernorm and pre_mlp_layernorm. "mla_up_proj": recompute the MLA
            up projection and RoPE applying parts."mlp": recompute the dense MLP layer."moe": recompute the
            MoE layer."moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing,
            "core_attn", "mlp", and "moe" uses normal checkpointing.
        clone_scatter_output_in_embedding (`bool`, *optional*, defaults to `True`):
            Argument clone_scatter_output_in_embedding. If not set, clone the output of the scatter in embedding
            layer to GC original tensor.
        profile (`bool`, *optional*, defaults to `False`):
            Argument profile. Enable nsys profiling. When using this option, nsys options should be specified
            in commandline. An example nsys commandline is `nsys profile -s none -t nvtx,cuda -o
            <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop`.
        profile_step_start (`int`, *optional*, defaults to 10):
            Argument profile_step_start. Global step to start profiling.
        profile_step_end (`int`, *optional*, defaults to 12):
            Argument profile_step_end. Global step to stop profiling.
        iterations_to_skip (`list[int]`, *optional*, defaults to []):
            Argument iterations_to_skip. List of iterations to skip, empty by default.
        result_rejected_tracker_filename (`str`):
            Argument result_rejected_tracker_filename. Optional name of file tracking `result_rejected` events.
        enable_gloo_process_groups (`bool`, *optional*, defaults to `True`):
            Argument enable_gloo_process_groups. Disables creation and usage of Gloo process groups.
        use_pytorch_profiler (`bool`, *optional*, defaults to `False`):
            Argument use_pytorch_profiler. Use the built-in pytorch profiler. Useful if you wish to view profiles
            in tensorboard.
        profile_ranks (`list[int]`, *optional*, defaults to [0]):
            Argument profile_ranks. Global ranks to profile.
        record_memory_history (`bool`, *optional*, defaults to `False`):
            Argument record_memory_history. Record memory history in last rank.
        memory_snapshot_path (`str`, *optional*, defaults to "snapshot.pickle"):
            Argument memory_snapshot_path. Specifies where to dump the memory history pickle.
        tp_comm_overlap (`bool`, *optional*, defaults to `False`):
            Argument tp_comm_overlap. Enables the  overlap of Tensor parallel communication and GEMM kernels.
        tp_comm_overlap_cfg (`str`):
            Argument tp_comm_overlap_cfg. Config file when tp_comm_overlap is enabled.
        tp_comm_overlap_ag (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_overlap_ag. Disables the All-Gather overlap with GEMM by pipelining the GEMM
            and All-Gather.
        tp_comm_overlap_rs (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_overlap_rs. Disables the Reduce-Scatter overlap with GEMM by pipelining the GEMM
            and Reduce-Scatter.
        tp_comm_overlap_rs_dgrad (`bool`, *optional*, defaults to `False`):
            Argument tp_comm_overlap_rs_dgrad. Enables the Reduce-Scatter overlap with dgrad GEMM.
        tp_comm_bulk_dgrad (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_bulk_dgrad. Disables the All-Gather overlap with bprop activation gradient GEMM.
        tp_comm_bulk_wgrad (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_bulk_wgrad. Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.
        tp_comm_bootstrap_backend (`str`, *optional*, defaults to "nccl"):
            Argument tp_comm_bootstrap_backend. Set the bootstrapping backend of Tensor parallel communications.
        use_cpu_initialization (`bool`):
            Argument use_cpu_initialization. If set, initialize weights on the CPU. This eliminates init
            differences based on tensor parallelism.
        empty_unused_memory_level (`int`, *optional*, defaults to 0):
            Argument empty_unused_memory_level. Call torch.cuda.empty_cache() each iteration (training and eval),
            to reduce fragmentation.0=off, 1=moderate, 2=aggressive.
        deterministic_mode (`bool`, *optional*, defaults to `False`):
            Argument deterministic_mode. Choose code that has deterministic execution. This usually means
            slower execution, but is good for debugging and testing.
        check_weight_hash_across_dp_replicas_interval (`int`):
            Argument check_weight_hash_across_dp_replicas_interval. Interval to check weight hashes are same across
            DP replicas. If not specified, weight hashes not checked.
        calculate_per_token_loss (`bool`, *optional*, defaults to `False`):
            Argument calculate_per_token_loss. Scale cross entropy loss by the number of non-padded tokens in
            the global batch, versus the default behavior of assuming all tokens are non-padded.
        train_sync_interval (`int`):
            Argument train_sync_interval. Training CPU-GPU synchronization interval, to ensure that CPU is not
            running too far ahead of GPU.
        checkpoint_activations (`bool`, *optional*, defaults to `False`):
            Argument checkpoint_activations. Checkpoint activation to allow for training with larger models,
            sequences, and batch sizes.
        train_iters (`int`):
            Argument train_iters. Total number of iterations to train over all training runs. Note that
            either train-iters or train-samples should be provided.
        train_samples (`int`):
            Argument train_samples. Total number of samples to train over all training runs. Note that
            either train-iters or train-samples should be provided.
        log_interval (`int`, *optional*, defaults to 100):
            Argument log_interval. Report loss and timing interval.
        exit_interval (`int`):
            Argument exit_interval. Exit the program after the iteration is divisible by this value.
        exit_duration_in_mins (`int`):
            Argument exit_duration_in_mins. Exit the program after this many minutes.
        exit_signal_handler (`bool`, *optional*, defaults to `False`):
            Argument exit_signal_handler. Dynamically save the checkpoint and shutdown the training if SIGTERM
            is received
        tensorboard_dir (`str`):
            Argument tensorboard_dir. Write TensorBoard logs to this directory.
        masked_softmax_fusion (`bool`, *optional*, defaults to `True`):
            Argument masked_softmax_fusion. Disable fusion of query_key_value scaling, masking, and softmax.
        bias_gelu_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_gelu_fusion. Disable bias and gelu fusion.
        bias_swiglu_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_swiglu_fusion. Disable bias and swiglu fusion, the fusion is available only when
            using megatron-core.
        bias_dropout_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_dropout_fusion. Disable bias and dropout fusion.
        apply_rope_fusion (`bool`, *optional*, defaults to `True`):
            Argument apply_rope_fusion. Disable rope fusion, the fusion is available only when using megatron-core.
        cross_entropy_loss_fusion (`bool`, *optional*, defaults to `False`):
            Argument cross_entropy_loss_fusion. Enabled fusion of cross entropy loss calculation.
        cross_entropy_fusion_impl (`str`, *optional*, defaults to "native"):
            Argument cross_entropy_fusion_impl. Implementation of cross entropy loss calculation.
        use_flash_attn (`bool`, *optional*, defaults to `False`):
            Argument use_flash_attn. use FlashAttention implementation of attention. https://arxiv.org/abs/2205.14135
        add_bias_linear (`bool`, *optional*, defaults to `True`):
            Argument add_bias_linear. Disable bias in the linear layers
        add_qkv_bias (`bool`, *optional*, defaults to `False`):
            Argument add_qkv_bias. Enable bias only in the QKV linear layers
        optimizer (`str`, *optional*, defaults to "adam"):
            Argument optimizer. Optimizer function
        optimizer_cpu_offload (`bool`, *optional*, defaults to `False`):
            Argument optimizer_cpu_offload. Offload optimizer state to CPU
        optimizer_offload_fraction (`float`, *optional*, defaults to 1.0):
            Argument optimizer_offload_fraction. Ratio of optimizer state to offload to CPU
        use_torch_optimizer_for_cpu_offload (`bool`, *optional*, defaults to `False`):
            Argument use_torch_optimizer_for_cpu_offload. Use torch.optim.Optimizer instead of Megatron's optimizer
            in optimizer cpu offload mode.
        overlap_cpu_optimizer_d2h_h2d (`bool`, *optional*, defaults to `False`):
            Argument overlap_cpu_optimizer_d2h_h2d. Overlap CPU optimizer step, gradients D2H and updated
            parameters H2D.
        pin_cpu_grads (`bool`, *optional*, defaults to `True`):
            Argument pin_cpu_grads. Disable pinning of CPU memory for gradients.
        pin_cpu_params (`bool`, *optional*, defaults to `True`):
            Argument pin_cpu_params. Disable pinning of CPU memory for parameters.
        dataloader_type (`str`):
            Argument dataloader_type. Single pass vs multiple pass data loader
        async_tensor_model_parallel_allreduce (`bool`, *optional*, defaults to `True`):
            Argument async_tensor_model_parallel_allreduce. DEPRECATED. This flag is ignored.
        no_persist_layer_norm (`bool`, *optional*, defaults to `False`):
            Argument no_persist_layer_norm. Disable using persistent fused layer norm kernel. This kernel supports
            only a set of hidden sizes. Please check persist_ln_hidden_sizes if your hidden size is supported.
        sequence_parallel (`bool`, *optional*, defaults to `False`):
            Argument sequence_parallel. Enable sequence parallel optimization.
        gradient_accumulation_fusion (`bool`, *optional*, defaults to `True`):
            Argument gradient_accumulation_fusion. Disable fusing gradient accumulation to weight gradient
            computation of linear layers
        deprecated_use_mcore_models (`bool`, *optional*, defaults to `False`):
            Argument deprecated_use_mcore_models. DEPRECATED. Use the implementation from megatron core.Now ignored
            and mcore models are the default, use --use-legacy-models to not use core models.
        use_legacy_models (`bool`, *optional*, defaults to `False`):
            Argument use_legacy_models. Use the legacy Megatron models, not Megatron-Core models.
        manual_gc (`bool`, *optional*, defaults to `False`):
            Argument manual_gc. Disable the threshold-based default garbage collector and trigger the
            garbage collection manually. Manual garbage collection helps to align the timing of the collection
            across ranks which mitigates the impact of CPU-associated jitters. When the manual gc is enabled,
            garbage collection is performed only at the start and the end of the validation routine by default.
        manual_gc_interval (`int`, *optional*, defaults to 0):
            Argument manual_gc_interval. Training step interval to trigger manual garbage collection. When the value
            is set to 0, garbage collection is not triggered between training steps.
        manual_gc_eval (`bool`, *optional*, defaults to `True`):
            Argument manual_gc_eval. When using manual garbage collection, disable garbage collection at the start
            and the end of each evaluation run.
        tp_comm_split_ag (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_split_ag. Disables the All-Gather overlap with fprop GEMM.
        tp_comm_split_rs (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_split_rs. Disables the Reduce-Scatter overlap with fprop GEMM.
        pipeline_model_parallel_comm_backend (`str`):
            Argument pipeline_model_parallel_comm_backend. Select a communicator backend for pipeline
            parallel communication. If None, the default backend will be used.
        high_priority_stream_groups (`list[str]`, *optional*, defaults to []):
            Argument high_priority_stream_groups. The communicator group names to use high priority streams.
        seed (`int`, *optional*, defaults to 1234):
            Argument seed. Random seed used for python, numpy, pytorch, and cuda.
        data_parallel_random_init (`bool`, *optional*, defaults to `False`):
            Argument data_parallel_random_init. Enable random initialization of params across data parallel ranks
        init_method_std (`float`, *optional*, defaults to 0.02):
            Argument init_method_std. Standard deviation of the zero mean normal distribution used for
            weight initialization.
        init_method_xavier_uniform (`bool`, *optional*, defaults to `False`):
            Argument init_method_xavier_uniform. Enable Xavier uniform parameter initialization
        lr (`float`):
            Argument lr. Initial learning rate. Depending on decay style and initial warmup, the learning rate at
            each iteration would be different.
        lr_decay_style (`str`, *optional*, defaults to "linear"):
            Argument lr_decay_style. Learning rate decay function.
        lr_wsd_decay_style (`str`, *optional*, defaults to "exponential"):
            Argument lr_wsd_decay_style. Decay style for the annealing phase of WSD
        lr_decay_iters (`int`):
            Argument lr_decay_iters. number of iterations to decay learning rate over, If None defaults
            to `--train-iters`
        lr_decay_samples (`int`):
            Argument lr_decay_samples. number of samples to decay learning rate over, If None defaults
            to `--train-samples`
        lr_wsd_decay_samples (`int`):
            Argument lr_wsd_decay_samples. number of samples for the annealing phase in the wsd schedule
        lr_wsd_decay_iters (`int`):
            Argument lr_wsd_decay_iters. number of iterations for the annealing phase in the wsd schedule
        lr_warmup_fraction (`float`):
            Argument lr_warmup_fraction. fraction of lr-warmup-(iters/samples) to use for warmup (as a float)
        lr_warmup_iters (`int`, *optional*, defaults to 0):
            Argument lr_warmup_iters. number of iterations to linearly warmup learning rate over.
        lr_warmup_samples (`int`, *optional*, defaults to 0):
            Argument lr_warmup_samples. number of samples to linearly warmup learning rate over.
        lr_warmup_init (`float`, *optional*, defaults to 0.0):
            Argument lr_warmup_init. Initial value for learning rate warmup. The scheduler starts warmup from
            this value.
        warmup (`int`):
            Argument warmup. Old lr warmup argument, do not use. Use one of the--lr-warmup-* arguments above
        min_lr (`float`, *optional*, defaults to 0.0):
            Argument min_lr. Minimum value for learning rate. The schedulerclip values below this threshold.
        override_opt_param_scheduler (`bool`, *optional*, defaults to `False`):
            Argument override_opt_param_scheduler. Reset the values of the scheduler (learning rate,warmup
            iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments
            and ignore values from checkpoints. Notethat all the above values will be reset.
        use_checkpoint_opt_param_scheduler (`bool`, *optional*, defaults to `False`):
            Argument use_checkpoint_opt_param_scheduler. Use checkpoint to set the values of the scheduler
            (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay
            style from checkpoint and ignore input arguments.
        decoupled_lr (`float`):
            Argument decoupled_lr. Separate learning rate for the input and output layer
        decoupled_min_lr (`float`):
            Argument decoupled_min_lr. Minimum value for learning rate for the input and output layer.
            The schedulerclip values below this threshold
        save (`str`):
            Argument save. Output directory to save checkpoints to.
        save_interval (`int`):
            Argument save_interval. Number of iterations between persistent checkpoint saves.
        no_save_optim (`bool`):
            Argument no_save_optim. Do not save current optimizer.
        no_save_rng (`bool`):
            Argument no_save_rng. Do not save current rng state.
        load (`str`):
            Argument load. Directory containing a model checkpoint.
        no_load_optim (`bool`):
            Argument no_load_optim. Do not load optimizer when loading checkpoint.
        no_load_rng (`bool`):
            Argument no_load_rng. Do not load rng state when loading checkpoint.
        non_persistent_save_interval (`int`):
            Argument non_persistent_save_interval. Number of iterations between non-persistent saves.
        non_persistent_ckpt_type (`str`):
            Argument non_persistent_ckpt_type. Type of non-persistent model checkpoints. "global" - Saved as a
            standard checkpoint (e.g., on Lustre) with old checkpoints being removed. "local" - Each rank saves
            a portion of the checkpoint locally (e.g., on SSD/ramdisk). None - No non-persistent checkpointing
            (default option).
        non_persistent_global_ckpt_dir (`str`):
            Argument non_persistent_global_ckpt_dir. Directory containing global non-persistent model checkpoints.
        non_persistent_local_ckpt_dir (`str`):
            Argument non_persistent_local_ckpt_dir. Directory containing local non-persistent model checkpoints.
        non_persistent_local_ckpt_algo (`str`, *optional*, defaults to "fully_parallel"):
            Argument non_persistent_local_ckpt_algo. Algorithm for local non-persistent checkpointing.
        finetune (`bool`, *optional*, defaults to `False`):
            Argument finetune. Load model for finetuning. Do not load optimizer or rng state from checkpoint and
            set iteration to 0. Assumed when loading a release checkpoint.
        pretrained_checkpoint (`str`):
            Argument pretrained_checkpoint. Directory containing a pretrained model checkpoint for finetuning.
        ckpt_step (`int`):
            Argument ckpt_step. Checkpoint step to load model from.
        perform_initialization (`bool`, *optional*, defaults to `True`):
            Argument perform_initialization. Do not perform initialization when building model, can reduce startup
            time when definitely loading from a checkpoint
        use_checkpoint_args (`bool`, *optional*, defaults to `False`):
            Argument use_checkpoint_args. Override model-related command-line arguments with arguments from checkpoint
        use_mp_args_from_checkpoint_args (`bool`, *optional*, defaults to `False`):
            Argument use_mp_args_from_checkpoint_args. Copy model parallelism command-line arguments from checkpoint
        use_tokenizer_model_from_checkpoint_args (`bool`, *optional*, defaults to `True`):
            Argument use_tokenizer_model_from_checkpoint_args. If set, do not use tokenizer model path from checkpoint
        exit_on_missing_checkpoint (`bool`, *optional*, defaults to `False`):
            Argument exit_on_missing_checkpoint. If '--load' is set, but checkpoint is not found (e.g., path
            typo), then exit instead of random initialization.
        use_dist_ckpt_deprecated (`bool`, *optional*, defaults to `False`):
            Argument use_dist_ckpt_deprecated. Deprecated: see --ckpt-format.
        use_persistent_ckpt_worker (`bool`, *optional*, defaults to `False`):
            Argument use_persistent_ckpt_worker. Enables a persitent checkpoint worker for async save
        auto_detect_ckpt_format (`bool`, *optional*, defaults to `False`):
            Argument auto_detect_ckpt_format. Determine if the checkpoint format is in legacy or distributed format.
            If False, expects distributed checkpoint iff args.ckpt_format != "torch". Might slow down loading a
            bit (double rank0 ckpt load).
        dist_ckpt_format_deprecated (`str`):
            Argument dist_ckpt_format_deprecated. Deprecated: see --ckpt-format.
        ckpt_format (`str`, *optional*, defaults to "torch_dist"):
            Argument ckpt_format. Checkpoint format to use. torch is the format used by torch.save/load. torch_dist
            is a megatron built-in distributed checkpointing format. torch_dcp is the
            torch.distributed.checkpoint format.
        ckpt_convert_format (`str`):
            Argument ckpt_convert_format. Checkpoint format for conversion.
        ckpt_convert_save (`str`):
            Argument ckpt_convert_save. Save directory for converted checkpoint.
        ckpt_convert_update_legacy_dist_opt_format (`bool`, *optional*, defaults to `False`):
            Argument ckpt_convert_update_legacy_dist_opt_format. When loading a checkpoint, update the legacy
            format for the distributed optimizer, which previously used a merged param/grad buffer and a
            different bucket mapping. The legacy format was deprecated on Feb 13, 2024.
        ckpt_fully_parallel_save_deprecated (`bool`, *optional*, defaults to `False`):
            Argument ckpt_fully_parallel_save_deprecated. Deprecated: see --no-ckpt-fully-parallel-save.
        ckpt_fully_parallel_save (`bool`, *optional*, defaults to `True`):
            Argument ckpt_fully_parallel_save. Disable applying full save parallelization across DP for
            distributed checkpoints. Depending on ckpt format might decrease the number of files in the
            checkpoint. Makes DistributedOptimizer checkpoint non-reshardable.
        async_save (`bool`):
            Argument async_save. Apply async checkpointing save. Currently works only with`torch_dist`
            distributed checkpoint format.
        ckpt_fully_parallel_load (`bool`, *optional*, defaults to `False`):
            Argument ckpt_fully_parallel_load. Apply full load parallelization across DP for distributed checkpoints.
        ckpt_assume_constant_structure (`bool`, *optional*, defaults to `False`):
            Argument ckpt_assume_constant_structure. If the model and optimizer state dict structure
            isconstant throughout a *single training job*, it allows fordifferent checkpointing
            performance optimizations.
        dist_ckpt_strictness (`str`, *optional*, defaults to "assume_ok_unexpected"):
            Argument dist_ckpt_strictness. Determine handling of key mismatch during checkpoint load.
            Check StrictHandling docs for flags meaning. NOTE: This flag controls only distributed checkpoint load
            from storage, not loading state dict into the model.
        load_model_opt_format (`bool`, *optional*, defaults to `False`):
            Argument load_model_opt_format. Load a checkpoint for TensorRT model optimizer
            (nvidia-modelopt).This function can also be used to load NeMo .nemo sharded checkpoints.
        fp16 (`bool`, *optional*, defaults to `False`):
            Argument fp16. Run model in fp16 mode.
        bf16 (`bool`, *optional*, defaults to `False`):
            Argument bf16. Run model in bfloat16 mode.
        grad_reduce_in_bf16 (`bool`, *optional*, defaults to `False`):
            Argument grad_reduce_in_bf16. Reduce gradients in bfloat16.
        loss_scale (`float`):
            Argument loss_scale. Static loss scaling, positive power of 2 values can improve fp16 convergence. If
            None, dynamicloss scaling is used.
        initial_loss_scale (`float`, *optional*, defaults to 4294967296):
            Argument initial_loss_scale. Initial loss-scale for dynamic loss scaling.
        min_loss_scale (`float`, *optional*, defaults to 1.0):
            Argument min_loss_scale. Minimum loss scale for dynamic loss scaling.
        loss_scale_window (`float`, *optional*, defaults to 1000):
            Argument loss_scale_window. Window over which to raise/lower dynamic scale.
        hysteresis (`int`, *optional*, defaults to 2):
            Argument hysteresis. hysteresis for dynamic loss scaling
        fp32_residual_connection (`bool`, *optional*, defaults to `False`):
            Argument fp32_residual_connection. Move residual connections to fp32.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `False`):
            Argument apply_query_key_layer_scaling. Scale Q * K^T by 1 / layer-number. Useful for fp16 training.
            Also sets `attention_softmax_in_fp32` to True.
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `False`):
            Argument attention_softmax_in_fp32. Run attention masking and softmax in fp32.
        accumulate_allreduce_grads_in_fp32 (`bool`, *optional*, defaults to `False`):
            Argument accumulate_allreduce_grads_in_fp32. Gradient accumulation and all-reduce in fp32.
        fp16_lm_cross_entropy (`bool`, *optional*, defaults to `False`):
            Argument fp16_lm_cross_entropy. Move the cross entropy unreduced loss calculationfor lm head to fp16.
        disable_bf16_reduced_precision_matmul (`bool`, *optional*, defaults to `False`):
            Argument disable_bf16_reduced_precision_matmul. If True,
            sets torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False to prevent matmul from
            using reduced precision accumulation when using BF16.
        reuse_grad_buf_for_mxfp8_param_ag (`bool`, *optional*, defaults to `False`):
            Argument reuse_grad_buf_for_mxfp8_param_ag. If True, reuse the grad buffer for MXFP8 parameter all-gather.
        tensor_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument tensor_model_parallel_size. Degree of tensor model parallelism.
        encoder_tensor_model_parallel_size (`int`, *optional*, defaults to 0):
            Argument encoder_tensor_model_parallel_size. DEPRECATED (will be removed in core_r0.14.0): Use
            orthotope parallelism management instead. Degree of tensor model parallelism for the encoder.
        pipeline_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument pipeline_model_parallel_size. Degree of pipeline model parallelism.
        encoder_pipeline_model_parallel_size (`int`, *optional*, defaults to 0):
            Argument encoder_pipeline_model_parallel_size. DEPRECATED (will be removed in core_r0.14.0): Use
            orthotope parallelism management instead. Degree of pipeline model parallelism in the encoder. This
            is independent of the amount of pipeline in the decoder.
        pipeline_model_parallel_split_rank (`int`):
            Argument pipeline_model_parallel_split_rank. Rank where encoder and decoder should be split.
            Deprecated; use --encoder-pipeline-model-parallel-size instead.
        decoder_first_pipeline_num_layers (`int`):
            Argument decoder_first_pipeline_num_layers. The number of transformer layers on the first pipeline stage
            of the decoder. Default None is even split of transformer layers across all pipeline stages
        decoder_last_pipeline_num_layers (`int`):
            Argument decoder_last_pipeline_num_layers. The number of transformer layers on the last pipeline stage
            of the decoder. Default None is even split of transformer layers across all pipeline stages
        pipeline_model_parallel_layout (`str`):
            Argument pipeline_model_parallel_layout. A string that describes a custom pipeline model parallel
            layout. e.g., "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer decoder layer, and
            mtp layer, respectively. Stages are split by "|". Replicated stages or layers can be described
            with multiplication. Commas can be used cosmetically. Default None is not using this argument to set
            the layout.
        model_parallel_size (`int`):
            Argument model_parallel_size. Old model parallel argument, do not use. Use
            --tensor-model-parallel-size instead.
        num_layers_per_virtual_pipeline_stage (`int`):
            Argument num_layers_per_virtual_pipeline_stage. Number of layers per virtual pipeline stage
        num_virtual_stages_per_pipeline_rank (`int`):
            Argument num_virtual_stages_per_pipeline_rank. Number of virtual pipeline stages per pipeline
            parallelism rank
        microbatch_group_size_per_vp_stage (`int`):
            Argument microbatch_group_size_per_vp_stage. Number of contiguous microbatches per virtual pipeline stage
        overlap_p2p_comm (`bool`, *optional*, defaults to `True`):
            Argument overlap_p2p_comm. overlap pipeline parallel communication with forward and backward chunks in 1F1B
        overlap_p2p_comm_warmup_flush (`bool`, *optional*, defaults to `False`):
            Argument overlap_p2p_comm_warmup_flush. if set, overlap pipeline parallel communication in warmup and flush
        distributed_backend (`str`, *optional*, defaults to "nccl"):
            Argument distributed_backend. Which backend to use for distributed training.
        distributed_timeout_minutes (`int`, *optional*, defaults to 10):
            Argument distributed_timeout_minutes. Timeout minutes for torch.distributed.
        overlap_grad_reduce (`bool`, *optional*, defaults to `False`):
            Argument overlap_grad_reduce. If set, overlap DDP grad reduce.
        defer_embedding_wgrad_compute (`bool`, *optional*, defaults to `False`):
            Argument defer_embedding_wgrad_compute. If set, defers the vocabulary projection linear
            layer weightgradient compute to pipeline flush.
        wgrad_deferral_limit (`int`, *optional*, defaults to 0):
            Argument wgrad_deferral_limit. Number of micro-batches for whichweight gradient computation of
            vocabulary projection is deferred, defaults to 0 whichmeans all the micro-batches are deferred. Invalid
            if `defer-embedding-wgrad-compute`is not set
        align_grad_reduce (`bool`, *optional*, defaults to `True`):
            Argument align_grad_reduce. If not set, all PP stages will launch gradient reduces
            simultaneously. Otherwise, each PP stage will independently launch as needed.
        ddp_num_buckets (`int`):
            Argument ddp_num_buckets. Number of buckets for data-parallel communication
        ddp_bucket_size (`int`):
            Argument ddp_bucket_size. Bucket size for data-parallel communication
        ddp_pad_buckets_for_high_nccl_busbw (`bool`, *optional*, defaults to `False`):
            Argument ddp_pad_buckets_for_high_nccl_busbw. If set, make sure the bucket size is divisible by a
            large power of 2 (2^16) to ensure NCCL collectives have high bus bandwidth at large DP counts, since
            NCCL message size (which for ring algorithms is bucket_size / dp_size) apparently needs to be divisible
            by a power of 2 for high busbw.
        ddp_average_in_collective (`bool`, *optional*, defaults to `False`):
            Argument ddp_average_in_collective. If set, average directly in data-parallel communication collective.
        overlap_param_gather (`bool`, *optional*, defaults to `False`):
            Argument overlap_param_gather. If set, overlap param all-gather in distributed optimizer.
        overlap_param_gather_with_optimizer_step (`bool`, *optional*, defaults to `False`):
            Argument overlap_param_gather_with_optimizer_step. If set, overlap param all-gather of first bucket
            with optimizer step.
        align_param_gather (`bool`, *optional*, defaults to `True`):
            Argument align_param_gather. If not set, all PP stages will launch param all-gathers
            simultaneously. Otherwise, each PP stage will independently launch as needed.
        scatter_gather_tensors_in_pipeline (`bool`, *optional*, defaults to `True`):
            Argument scatter_gather_tensors_in_pipeline. If not set, use scatter/gather to optimize communication
            of tensors in pipeline.
        use_ring_exchange_p2p (`bool`, *optional*, defaults to `False`):
            Argument use_ring_exchange_p2p. If set, use custom-built ring exchange for p2p communications. Note
            that this option will require a custom built image that support ring-exchange p2p.
        local_rank (`int`, *optional*, defaults to 0):
            Argument local_rank. local rank passed from distributed launcher.
        lazy_mpu_init (`bool`):
            Argument lazy_mpu_init. If set to True, initialize_megatron() skips DDP initialization and returns
            function to complete it instead. Also turns on --use-cpu-initialization flag. This is for external
            DDP manager.
        account_for_embedding_in_pipeline_split (`bool`, *optional*, defaults to `False`):
            Argument account_for_embedding_in_pipeline_split. If set, *input* embedding layer will be treated as
            a standard transformerlayer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (`bool`, *optional*, defaults to `False`):
            Argument account_for_loss_in_pipeline_split. If set, loss layer will be treated as a
            standard transformerlayer in the context of partition and placement for pipeline parallelism.
        use_distributed_optimizer (`bool`, *optional*, defaults to `False`):
            Argument use_distributed_optimizer. Use distributed optimizer.
        nccl_ub (`bool`, *optional*, defaults to `False`):
            Argument nccl_ub. Use the userbuffer registration for DP/FSDP communication buffers.This option will
            reduce GPU SM usage for the DP/FSDP communication,which is improving the performance of the
            overlapped computation.
        use_sharp (`bool`, *optional*, defaults to `False`):
            Argument use_sharp. Required to enable SHARP communication.
        use_custom_fsdp (`bool`, *optional*, defaults to `False`):
            Argument use_custom_fsdp. Use the Megatron FSDP code path in DDP.
        init_model_with_meta_device (`bool`, *optional*, defaults to `False`):
            Argument init_model_with_meta_device.
        data_parallel_sharding_strategy (`str`, *optional*, defaults to "no_shard"):
            Argument data_parallel_sharding_strategy. Sharding strategy of data parallelism.
        gradient_reduce_div_fusion (`bool`, *optional*, defaults to `True`):
            Argument gradient_reduce_div_fusion. If not set, fuse the division in gradient reduce.
        fsdp_double_buffer (`bool`, *optional*, defaults to `False`):
            Argument fsdp_double_buffer. Enable double buffering for temporary memory needed for custom
            FSDP communications. Double-buffering the communication memory improves memory management efficiency
            by reusing previously allocated buffers, rather than creating new buffers for each FSDP communication.
            This is required for user buffer registration and is enabled by default when using NCCL user buffers.
        suggested_communication_unit_size (`int`):
            Argument suggested_communication_unit_size. Specifies the number of elements to communicate at once
            during FSDP (Fully Sharded Data Parallel) operations. This flag also affects FSDP all-gather
            prefetch behavior. Setting a larger value increases the communication buffer size, while a smaller
            value disables prefetching and may degrade performance. Adjust this value based on your system's memory
            and performance requirements.
        keep_fp8_transpose_cache_when_using_custom_fsdp (`bool`, *optional*, defaults to `False`):
            Argument keep_fp8_transpose_cache_when_using_custom_fsdp. If set, keep the fp8 transpose cache when
            using custom FSDP.
        num_distributed_optimizer_instances (`int`, *optional*, defaults to 1):
            Argument num_distributed_optimizer_instances. Number of Distributed Optimizer copies across Data
            Parallel domain.
        use_torch_fsdp2 (`bool`, *optional*, defaults to `False`):
            Argument use_torch_fsdp2. Use the torch FSDP2 implementation. FSDP2 has not been tested with
            pipeline parallelism, and may contain bugs.
        torch_fsdp2_reshard_after_forward (`bool`, *optional*, defaults to `True`):
            Argument torch_fsdp2_reshard_after_forward. Whether to reshard weights after forward pass when
            using PyTorch FSDP2. Set to enable FSDP ZeRO-2.
        context_parallel_size (`int`, *optional*, defaults to 1):
            Argument context_parallel_size. Degree of context parallelism.
        cp_comm_type (`list[str]`, *optional*, defaults to ['p2p']):
            Argument cp_comm_type. Inter-gpu communication type for context parallelism: p2p, a2a, allgather
            or a2a+p2p. If a single string is provided, all layers will share the same communication type. Users
            can also specify separated types for each layer like --cp-comm-type p2p p2p a2a a2a a2a+p2p a2a+p2p
        hierarchical_context_parallel_sizes (`list[int]`):
            Argument hierarchical_context_parallel_sizes. Degrees of the hierarchical context parallelism. Users
            should provide a list to specify the sizes for different levels. --hierarchical-context-parallel-sizes 2
            4 indicates every two adjacent gpus forms the first level of cp groups and the cp ranks with the
            same odevity forms the second level of cp groups.
        nccl_communicator_config_path (`str`):
            Argument nccl_communicator_config_path. Path to the yaml file with NCCL communicator configurations.
            The number of min/max thread groups and thread group cluster size of each communicator can be configured
            by setting `min_ctas`, `max_ctas`, and `cga_cluster_size`.
        use_tp_pp_dp_mapping (`bool`, *optional*, defaults to `False`):
            Argument use_tp_pp_dp_mapping. If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp
            to tp-cp-ep-pp-dp.
        replication (`bool`, *optional*, defaults to `False`):
            Argument replication. If set, replication of local checkpoints is enabled. Needs to be enabled on all ranks.
        replication_jump (`int`):
            Argument replication_jump. Specifies `J`, the spacing between ranks storing replicas of a given
            rank's data. Replicas for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. This flag
            has an effect only if --replication is used. and must be consistent across all ranks.
        replication_factor (`int`, *optional*, defaults to 2):
            Argument replication_factor. Number of machines storing the replica of a given rank's data.
        eval_iters (`int`, *optional*, defaults to 100):
            Argument eval_iters. Number of iterations to run for evaluationvalidation/test for.
        eval_interval (`int`, *optional*, defaults to 1000):
            Argument eval_interval. Interval between running evaluation on validation set.
        test_mode (`bool`, *optional*, defaults to `False`):
            Argument test_mode. Run all real-time test alongside the experiment.
        skip_train (`bool`, *optional*, defaults to `False`):
            Argument skip_train. If set, bypass the training loop, optionally do evaluation for validation/test,
            and exit.
        data_path (`list[str]`):
            Argument data_path. The weight and prefix list for a set of train, validation, and testdatasets which
            split according to --split. The accepted formats are: (1) a single prefix, (2) a list of weight
            prefix pairs e.g. weight1 prefix1 weight2 prefix2, (3) a list of prefixes e.g. prefix1 prefix2. For
            (3), weights are inferred from the lengths of the contributing datasets. This argument is exclusive to
            the other independent --*-data-path arguments.
        split (`str`):
            Argument split. Comma-separated list of proportions for training, validation, and test split. For
            example the split `90,5,5` will use 90%% of data for training, 5%% for validation and 5%% for test.
        train_data_path (`list[str]`):
            Argument train_data_path. The weight and prefix list for an independent train dataset. Follows the
            same pattern rules as --data-path.
        valid_data_path (`list[str]`):
            Argument valid_data_path. The weight and prefix list for an independent validation dataset. Follows
            the same pattern rules as --data-path.
        test_data_path (`list[str]`):
            Argument test_data_path. The weight and prefix list for an independent test dataset. Follows the
            same pattern rules as --data-path.
        data_args_path (`str`):
            Argument data_args_path. Path to data-args. Instead of feeding `--data-path` with weighted dataset, we
            pass in a file path from which we read that argument. This is useful when the list of data is too big.
        per_split_data_args_path (`str`):
            Argument per_split_data_args_path. Path to per-split-data-args. Instead of
            feeding `--(train|valid|test)-data-path` with weighted dataset, we pass in a file path from which we
            read those arguments. This is useful when the list of data is too big. Format is a json file with
            `train`, `valid, `test` keys
        data_cache_path (`str`):
            Argument data_cache_path. Path to a directory to hold cached index files.
        mmap_bin_files (`bool`, *optional*, defaults to `True`):
            Argument mmap_bin_files. Disable mmap-ing of .bin files.
        mock_data (`bool`, *optional*, defaults to `False`):
            Argument mock_data. Skip data loading and validation and opt for artificial generation of mock data when
            an implementation is available.
        seq_length (`int`):
            Argument seq_length. Maximum sequence length to process.
        encoder_seq_length (`int`):
            Argument encoder_seq_length. Maximum encoder sequence length to process.This should be exclusive
            of --seq-length
        decoder_seq_length (`int`):
            Argument decoder_seq_length. Maximum decoder sequence length to process.
        retriever_seq_length (`int`, *optional*, defaults to 256):
            Argument retriever_seq_length. Maximum sequence length for the biencoder model for retriever
        sample_rate (`float`, *optional*, defaults to 1.0):
            Argument sample_rate. sample rate for training data. Supposed to be 0  < sample_rate < 1
        mask_prob (`float`, *optional*, defaults to 0.15):
            Argument mask_prob. Probability of replacing a token with mask.
        short_seq_prob (`float`, *optional*, defaults to 0.1):
            Argument short_seq_prob. Probability of producing a short sequence.
        num_workers (`int`, *optional*, defaults to 2):
            Argument num_workers. Dataloader number of workers.
        reset_position_ids (`bool`, *optional*, defaults to `False`):
            Argument reset_position_ids. Reset posistion ids after end-of-document token.
        reset_attention_mask (`bool`, *optional*, defaults to `False`):
            Argument reset_attention_mask. Reset self attention maske after end-of-document token.
        eod_mask_loss (`bool`, *optional*, defaults to `False`):
            Argument eod_mask_loss. Mask loss for the end of document tokens.
        create_attention_mask_in_dataloader (`bool`, *optional*, defaults to `True`):
            Argument create_attention_mask_in_dataloader. If set, do not create attention_masks in dataloader.
        num_dataset_builder_threads (`int`, *optional*, defaults to 1):
            Argument num_dataset_builder_threads. Number of parallel threads per rank for dataset builder
        object_storage_cache_path (`str`):
            Argument object_storage_cache_path. Path to cache index files when using s3 or msc dataloader
        mid_level_dataset_surplus (`float`, *optional*, defaults to 0.005):
            Argument mid_level_dataset_surplus. The sample surplus to build for the mid-level datasets(s)
        vocab_size (`int`):
            Argument vocab_size. Size of vocab before EOD or padding.
        vocab_file (`str`):
            Argument vocab_file. Path to the vocab file.
        merge_file (`str`):
            Argument merge_file. Path to the BPE merge file.
        vocab_extra_ids (`int`, *optional*, defaults to 0):
            Argument vocab_extra_ids. Number of additional vocabulary tokens. They are used for span masking in the
            T5 model
        tokenizer_type (`str`):
            Argument tokenizer_type. What type of tokenizer to use.
        tokenizer_model (`str`):
            Argument tokenizer_model. Sentencepiece tokenizer model.
        tiktoken_pattern (`str`):
            Argument tiktoken_pattern. Which tiktoken pattern to use. Options: [v1, v2]
        tiktoken_num_special_tokens (`int`, *optional*, defaults to 1000):
            Argument tiktoken_num_special_tokens. Number of special tokens in tiktoken tokenizer
        tiktoken_special_tokens (`list[str]`):
            Argument tiktoken_special_tokens. List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]
        adlr_autoresume (`bool`, *optional*, defaults to `False`):
            Argument adlr_autoresume. Enable autoresume on adlr cluster.
        adlr_autoresume_interval (`int`, *optional*, defaults to 1000):
            Argument adlr_autoresume_interval. Intervals over which check for autoresumetermination signal
        ict_head_size (`int`):
            Argument ict_head_size. Size of block embeddings to be used in ICT and REALM (paper default: 128)
        biencoder_projection_dim (`int`, *optional*, defaults to 0):
            Argument biencoder_projection_dim. Size of projection head used in biencoder (paper default: 128)
        biencoder_shared_query_context_model (`bool`, *optional*, defaults to `False`):
            Argument biencoder_shared_query_context_model. Whether to share the parameters of the query and
            context models or not
        ict_load (`str`):
            Argument ict_load. Directory containing an ICTBertModel checkpoint
        bert_load (`str`):
            Argument bert_load. Directory containing an BertModel checkpoint (needed to start ICT and REALM)
        titles_data_path (`str`):
            Argument titles_data_path. Path to titles dataset used for ICT
        query_in_block_prob (`float`, *optional*, defaults to 0.1):
            Argument query_in_block_prob. Probability of keeping query in block for ICT dataset
        use_one_sent_docs (`bool`, *optional*, defaults to `False`):
            Argument use_one_sent_docs. Whether to use one sentence documents in ICT
        evidence_data_path (`str`):
            Argument evidence_data_path. Path to Wikipedia Evidence frm DPR paper
        retriever_report_topk_accuracies (`list[int]`, *optional*, defaults to []):
            Argument retriever_report_topk_accuracies. Which top-k accuracies to report (e.g. '1 5 20')
        retriever_score_scaling (`bool`, *optional*, defaults to `False`):
            Argument retriever_score_scaling. Whether to scale retriever scores by inverse square root of hidden size
        block_data_path (`str`):
            Argument block_data_path. Where to save/load BlockData to/from
        embedding_path (`str`):
            Argument embedding_path. Where to save/load Open-Retrieval Embedding data to/from
        indexer_batch_size (`int`, *optional*, defaults to 128):
            Argument indexer_batch_size. How large of batches to use when doing indexing jobs
        indexer_log_interval (`int`, *optional*, defaults to 1000):
            Argument indexer_log_interval. After how many batches should the indexer report progress
        num_classes (`int`, *optional*, defaults to 1000):
            Argument num_classes. num of classes in vision classificaiton task
        img_h (`int`, *optional*, defaults to 224):
            Argument img_h. Image height for vision classification task
        img_w (`int`, *optional*, defaults to 224):
            Argument img_w. Image height for vision classification task
        num_channels (`int`, *optional*, defaults to 3):
            Argument num_channels. Number of channels in input image data
        patch_dim (`int`, *optional*, defaults to 16):
            Argument patch_dim. patch dimension
        classes_fraction (`float`, *optional*, defaults to 1.0):
            Argument classes_fraction. training with fraction of classes.
        data_per_class_fraction (`float`, *optional*, defaults to 1.0):
            Argument data_per_class_fraction. training with fraction of data per class.
        data_sharding (`bool`, *optional*, defaults to `True`):
            Argument data_sharding. Disable data sharding.
        head_lr_mult (`float`, *optional*, defaults to 1.0):
            Argument head_lr_mult. learning rate multiplier for head during finetuning
        vision_pretraining (`bool`, *optional*, defaults to `False`):
            Argument vision_pretraining. flag to indicate vision pretraining
        vision_pretraining_type (`str`, *optional*, defaults to "classify"):
            Argument vision_pretraining_type. pretraining objectives
        vision_backbone_type (`str`, *optional*, defaults to "vit"):
            Argument vision_backbone_type. backbone types types
        swin_backbone_type (`str`, *optional*, defaults to "tiny"):
            Argument swin_backbone_type. pretraining objectives
        mask_type (`str`, *optional*, defaults to "random"):
            Argument mask_type. mask types
        mask_factor (`float`, *optional*, defaults to 1.0):
            Argument mask_factor. mask size scaling parameter
        iter_per_epoch (`int`, *optional*, defaults to 1250):
            Argument iter_per_epoch. iterations per epoch
        dino_local_img_size (`int`, *optional*, defaults to 96):
            Argument dino_local_img_size. Image size for vision classification task
        dino_local_crops_number (`int`, *optional*, defaults to 10):
            Argument dino_local_crops_number. Number of local crops
        dino_head_hidden_size (`int`, *optional*, defaults to 2048):
            Argument dino_head_hidden_size. Hidden dimension size in dino head
        dino_bottleneck_size (`int`, *optional*, defaults to 256):
            Argument dino_bottleneck_size. Bottle neck dimension in dino head
        dino_freeze_last_layer (`float`, *optional*, defaults to 1):
            Argument dino_freeze_last_layer. Freezing last layer weights
        dino_norm_last_layer (`bool`, *optional*, defaults to `False`):
            Argument dino_norm_last_layer. Disable Norm in last layer.
        dino_warmup_teacher_temp (`float`, *optional*, defaults to 0.04):
            Argument dino_warmup_teacher_temp. warump teacher temperature
        dino_teacher_temp (`float`, *optional*, defaults to 0.07):
            Argument dino_teacher_temp. teacher temperature
        dino_warmup_teacher_temp_epochs (`int`, *optional*, defaults to 30):
            Argument dino_warmup_teacher_temp_epochs. warmup teacher temperaure epochs
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Argument qk_layernorm. Whether to layer normalize the q and k attention embeddings.
        qk_l2_norm (`bool`, *optional*, defaults to `False`):
            Argument qk_l2_norm. Use llama 4 qk l2 norm
        expert_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument expert_model_parallel_size. Degree of expert model parallelism.
        expert_tensor_parallel_size (`int`):
            Argument expert_tensor_parallel_size. Degree of expert model parallelism. Default is None, which will
            be set to the value of --tensor-model-paralle-size.
        num_experts (`int`):
            Argument num_experts. Number of Experts in MoE (None means no MoE)
        moe_layer_freq (`Any`, *optional*, defaults to 1):
            Argument moe_layer_freq. Frequency between MoE layers and Dense layers. Accepts either: - An integer
            N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers - A string containing
            a Python list expression that defines a custom pattern, e.g.: "([1]*3+[0]*1)*3" evaluates
            to [1,1,1,0,1,1,1,0,1,1,1,0] where 1 indicates an expert layer and 0 indicates a dense layer.
            Examples: "([0]+[1]*23)": 1 dense layer followed by 23 experts layers, "([1]*3+[0]*2)*2": Three
            expert layers followed by two dense layers, repeated twice.
        moe_ffn_hidden_size (`int`):
            Argument moe_ffn_hidden_size. The hidden size of each expert's feed-forward network (ffn). If
            not specified, defaults to the ffn_hidden_size.
        moe_shared_expert_intermediate_size (`int`):
            Argument moe_shared_expert_intermediate_size. Shared expert total ffn hidden size. It should be equal
            to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. None
            means no shared expert.
        moe_shared_expert_overlap (`bool`, *optional*, defaults to `False`):
            Argument moe_shared_expert_overlap. Enable overlapping between shared expert computations and
            dispatcher communications. Without this, the shared epxerts execute after the routed experts.
            Only effective when moe-shared-expert-intermediate-size is set.
        moe_grouped_gemm (`bool`, *optional*, defaults to `False`):
            Argument moe_grouped_gemm. When there are multiple experts per rank, launch multiple local GEMM kernels
            in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine.
        moe_use_legacy_grouped_gemm (`bool`, *optional*, defaults to `False`):
            Argument moe_use_legacy_grouped_gemm. Use legacy GroupedMLP rather than TEGroupedMLP. Note: The legacy
            one will be deprecated soon.
        moe_layer_recompute (`bool`, *optional*, defaults to `False`):
            Argument moe_layer_recompute. Enable checkpointing for moe_layer, should be used when memory is
            not sufficient. Deprecated. Use "--recompute-granularity selective --recompute-modules moe" instead.
        moe_extended_tp (`bool`, *optional*, defaults to `False`):
            Argument moe_extended_tp. Deprecated. Use --expert-tensor-parallel-size instead.
        moe_use_upcycling (`bool`, *optional*, defaults to `False`):
            Argument moe_use_upcycling. Load a checkpoint of a dense model, convert it into an MoE model, and save
            the converted model to the path specified by --save. Upcycling is implemented on the top of
            distributed checkpointing, so it supports parallel modes different from the dense model.
        moe_router_load_balancing_type (`str`, *optional*, defaults to "aux_loss"):
            Argument moe_router_load_balancing_type. Determines the load balancing strategy for the router.
            "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer;
            "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2, which computes the loss for
            each individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and
            "none" implies no load balancing. The default is "aux_loss".
        moe_router_dtype (`str`):
            Argument moe_router_dtype. Data type for routing computation and expert output weighted
            averaging. Fp32/fp64 enhances numerical stability, especially with numerous experts. The perf impact
            should be negligible when used with permute fusion. None means no changes for dtype.
        moe_router_score_function (`str`, *optional*, defaults to "softmax"):
            Argument moe_router_score_function. Score function for MoE TopK routing. Can be "softmax" or "sigmoid".
        moe_router_topk (`int`, *optional*, defaults to 2):
            Argument moe_router_topk. Number of experts to route to for each token. The default is 2.
        moe_router_pre_softmax (`bool`, *optional*, defaults to `False`):
            Argument moe_router_pre_softmax. Enable pre-softmax routing for MoE, which means softmax is before
            the top-k selection. By default, softmax is done after top-k.
        moe_router_num_groups (`int`):
            Argument moe_router_num_groups. Number of groups to divide experts into for group-limited routing.
            When using group-limited routing: 1) Experts are divided into equal-sized groups, 2) For each token,
            a subset of groups are selected based on routing scores (sum of top-2 expert scores within each group),
            3) From these selected groups, moe_router_topk experts are chosen.Two common use cases: 1)
            Device-limited routing: Set equal to expert parallel size (EP) to limit each token to experts on a
            subset of devices (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434) 2) Node-limited routing: Set equal
            to number of nodes in EP group to limit each token to experts on a subset of nodes (See
            DeepSeek-V3: https://arxiv.org/pdf/2412.19437)
        moe_router_group_topk (`int`):
            Argument moe_router_group_topk. Number of selected groups for group-limited routing.
        moe_router_topk_scaling_factor (`float`):
            Argument moe_router_topk_scaling_factor. Scaling factor for routing score in top-k selection, only
            works when --moe-router-pre-softmax enabled. Defaults to None, which means no scaling.
        moe_router_enable_expert_bias (`bool`, *optional*, defaults to `False`):
            Argument moe_router_enable_expert_bias. TopK routing with dynamic expert bias in the aux-loss-free
            load balancing strategy. The routing decision is based on the sum of the routing scores and the
            expert bias. See https://arxiv.org/abs/2408.15664 for details.
        moe_router_bias_update_rate (`float`, *optional*, defaults to 0.001):
            Argument moe_router_bias_update_rate. Expert bias update rate in the aux-loss-free load balancing
            strategy. The expert bias is updated based on the number of assigned tokens to each expert in a
            global batch, where the bias is increased for the experts with less assigned tokens and decreased for
            the experts with more assigned tokens. The default value 1e-3 is same as that used in DeepSeekV3.
        moe_router_force_load_balancing (`bool`, *optional*, defaults to `False`):
            Argument moe_router_force_load_balancing. [Experimental] Force override routing to balance
            token distribution using random logits for MoE routers, supporting naive top-k and group-limited
            top-k. This experimental feature is for benchmarking purposes only!
        moe_router_padding_for_fp8 (`bool`, *optional*, defaults to `False`):
            Argument moe_router_padding_for_fp8. Pad the routing_map to make sure the number of tokens each
            expert received is a multiple of 16/32 for FP8 precision. It is suggested to enable this for
            dropless training with FP8 precision when num_local_experts > 1. This is a more efficient way to pad
            for FP8 which eliminates the explicit padding in the GroupedMLP layer.
        moe_aux_loss_coeff (`float`, *optional*, defaults to 0.0):
            Argument moe_aux_loss_coeff. Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.
        moe_z_loss_coeff (`float`):
            Argument moe_z_loss_coeff. Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.
        moe_input_jitter_eps (`float`):
            Argument moe_input_jitter_eps. Add noise to the input tensor by applying jitter with a specified
            epsilon value.
        moe_per_layer_logging (`bool`, *optional*, defaults to `False`):
            Argument moe_per_layer_logging. Enable per-layer logging for MoE, currently supports auxiliary loss and
            z loss.
        moe_token_dispatcher_type (`str`, *optional*, defaults to "allgather"):
            Argument moe_token_dispatcher_type. The type of token dispatcher to use. The default is
            'allgather'. Options are 'allgather', 'alltoall'. We recommend using 'alltoall' when applying
            expert parallelism. For more information, please refer to the documentation in core/moe/README.
        moe_enable_deepep (`bool`, *optional*, defaults to `False`):
            Argument moe_enable_deepep. [Experimental] Enable DeepSeek/DeepEP for efficient token dispatching
            and combine in MoE models. Only works with flex token dispatcher by
            setting --moe-token-dispatcher-type=flex.
        moe_deepep_num_sms (`int`, *optional*, defaults to 20):
            Argument moe_deepep_num_sms. Number of SMs to use for DeepEP.
        moe_permute_fusion (`bool`, *optional*, defaults to `False`):
            Argument moe_permute_fusion. Fuse token rearrangement ops during token dispatching.
        moe_expert_capacity_factor (`float`):
            Argument moe_expert_capacity_factor. The capacity factor for each expert, None means no token will
            be dropped.
        moe_pad_expert_input_to_capacity (`bool`, *optional*, defaults to `False`):
            Argument moe_pad_expert_input_to_capacity. Pads the input for each expert to match the expert
            capacity length, effective only after the --moe-expert-capacity-factor is set.
        moe_token_drop_policy (`str`, *optional*, defaults to "probs"):
            Argument moe_token_drop_policy. The policy to drop tokens. Can be either "probs" or "position". If
            "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of
            each batch will be dropped.
        moe_apply_probs_on_input (`bool`, *optional*, defaults to `False`):
            Argument moe_apply_probs_on_input. Apply probs before mlp activation for moe routing.
        delay_wgrad_compute (`bool`, *optional*, defaults to `False`):
            Argument delay_wgrad_compute. Delay the wgrad compute for batch-level overlapping
        moe_upcycling_granularity (`int`, *optional*, defaults to 1):
            Argument moe_upcycling_granularity. This param sepecifics how many times smaller is the expert hidden
            size compared with the original dense FFN hidden size. For using granular upcycling strategy, please
            set this param as a positive integer. If this param is set to 1, it means using the default
            upcycling strategy.
        q_lora_rank (`int`):
            Argument q_lora_rank. Rank of Query tensor's low rank representation.
        kv_lora_rank (`int`, *optional*, defaults to 32):
            Argument kv_lora_rank. Rank of Key and Value tensors' low rank representation.
        qk_head_dim (`int`, *optional*, defaults to 128):
            Argument qk_head_dim. Dimension of the head in the QK projection. q_head_dim = qk_head_dim
            + qk_pos_emb_head_dim
        qk_pos_emb_head_dim (`int`, *optional*, defaults to 64):
            Argument qk_pos_emb_head_dim. Dimension of the position embedding in the QK projection.
        v_head_dim (`int`, *optional*, defaults to 128):
            Argument v_head_dim. Dimension of the head in the V projection.
        rotary_scaling_factor (`float`, *optional*, defaults to 1.0):
            Argument rotary_scaling_factor. Rotary scaling factor for the rotary embeddings.
        mscale (`float`, *optional*, defaults to 1.0):
            Argument mscale. Mscale for YaRN RoPE in multi-latent attention.
        mscale_all_dim (`float`, *optional*, defaults to 1.0):
            Argument mscale_all_dim. Mscale all dimensions for YaRN RoPE in multi-latent attention.
        heterogeneous_layers_config_path (`str`):
            Argument heterogeneous_layers_config_path. Path to json file containing heterogeneous model
            configuration. Use the format of the HuggingFace config files in llama nemotron models,
            e.g. https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json.
        heterogeneous_layers_config_encoded_json (`str`):
            Argument heterogeneous_layers_config_encoded_json. This is encoded json string of the heterogeneous
            model configuration. Used to keep the content of the heterogeneous model specification in args when
            the model is loaded from a checkpoint. Use the format of the HuggingFace config files in llama
            nemotron models,
            e.g. https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/resolve/main/config.json.
        log_params_norm (`bool`, *optional*, defaults to `False`):
            Argument log_params_norm. If set, calculate and log parameters norm.
        log_num_zeros_in_grad (`bool`, *optional*, defaults to `False`):
            Argument log_num_zeros_in_grad. If set, calculate and log the number of zeros in gradient.
        log_throughput (`bool`, *optional*, defaults to `False`):
            Argument log_throughput. If set, calculate and log throughput per GPU.
        log_progress (`bool`, *optional*, defaults to `False`):
            Argument log_progress. If set, log progress (in terms of number of processed tokens and number
            of floating-point operations) to progress.txt file in checkpoint directory.
        timing_log_level (`int`, *optional*, defaults to 0):
            Argument timing_log_level. Granularity level to measure and report timing.    0: report only iteration
            time and make sure timing       does not introduce extra overhead.   1: report timing for operations
            that are executed       very limited times (basically once) during       each iteration (such as
            gradient all-reduce)    2: report timing for operations that migh be       executed numerous times
            during each iteration. Note that setting the level to 1 or 2 might cause increase in iteration time.
        log_energy (`bool`, *optional*, defaults to `False`):
            Argument log_energy. If set, log energy consumption (in Joules)
        barrier_with_L1_time (`bool`, *optional*, defaults to `True`):
            Argument barrier_with_L1_time. If not set, use barrier with level 1 time measurements. Note that this is
            up to the user to make sure calling barrier with their timers will not result in hangs. This can happen
            if for example the user adds a level 1 timer that is not called by all ranks.
        timing_log_option (`str`, *optional*, defaults to "minmax"):
            Argument timing_log_option. Options for logging timing:  max: report the max timing across all ranks
             minmax: report min and max timings across all ranks  all: report timings of all ranks.
        tensorboard_log_interval (`int`, *optional*, defaults to 1):
            Argument tensorboard_log_interval. Report to tensorboard interval.
        tensorboard_queue_size (`int`, *optional*, defaults to 1000):
            Argument tensorboard_queue_size. Size of the tensorboard queue for pending events and summaries before
            one of the ‘add’ calls forces a flush to disk.
        log_timers_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_timers_to_tensorboard. If set, write timers to tensorboard.
        log_loss_scale_to_tensorboard (`bool`, *optional*, defaults to `True`):
            Argument log_loss_scale_to_tensorboard. Disable loss-scale logging to tensorboard.
        log_validation_ppl_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_validation_ppl_to_tensorboard. If set, write validation perplexity to tensorboard.
        log_memory_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_memory_to_tensorboard. Enable memory logging to tensorboard.
        log_world_size_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_world_size_to_tensorboard. Enable world size logging to tensorboard.
        wandb_project (`str`, *optional*, defaults to ""):
            Argument wandb_project. The wandb project name. Ignore wandb by default.
        wandb_exp_name (`str`, *optional*, defaults to ""):
            Argument wandb_exp_name. The wandb experiment name.
        wandb_save_dir (`str`, *optional*, defaults to ""):
            Argument wandb_save_dir. Path to save the wandb results locally.
        logging_level (`int`):
            Argument logging_level. Set default logging level
        log_straggler (`bool`, *optional*, defaults to `False`):
            Argument log_straggler. If set, tracks and logs straggler per GPU.
        disable_straggler_on_startup (`bool`, *optional*, defaults to `False`):
            Argument disable_straggler_on_startup. If set, StragglerDetector is disabled on startup.
        straggler_ctrlr_port (`int`, *optional*, defaults to 65535):
            Argument straggler_ctrlr_port. Port number to toggle StragglerDetector on/off at runtime
        straggler_minmax_count (`int`, *optional*, defaults to 1):
            Argument straggler_minmax_count. Number of ranks to report with high/low estimated throughput
        run_workload_inspector_server (`bool`, *optional*, defaults to `False`):
            Argument run_workload_inspector_server. If set, enables workload inspector server for on-demand profiling.
        inference_batch_times_seqlen_threshold (`int`, *optional*, defaults to -1):
            Argument inference_batch_times_seqlen_threshold. If (batch-size * sequence-length) is smaller than
            this thresholdthen batches will not be split up for pipelining.Requires
            setting --pipeline-model-parallel-size > 1.Setting this to -1 indicates that batch pipelining is not used.
        max_tokens_to_oom (`int`, *optional*, defaults to 12000):
            Argument max_tokens_to_oom. Maximum number of tokens during inferencetokens here is # in prompt + #
            to generateAllows us to throw an error before OOM crashes server
        output_bert_embeddings (`bool`, *optional*, defaults to `False`):
            Argument output_bert_embeddings. Output Bert embeddings (via mean pooling) from model, rather than
            its binary head output or entire hidden batch.
        bert_embedder_type (`str`, *optional*, defaults to "megatron"):
            Argument bert_embedder_type. Select either Megatron or Huggingface as the Bert embedder.
        flash_decode (`bool`, *optional*, defaults to `False`):
            Argument flash_decode. Whether to use the flash decoding kernel.
        enable_cuda_graph (`bool`, *optional*, defaults to `False`):
            Argument enable_cuda_graph. Use CUDA graph capture and replay.
        cuda_graph_warmup_steps (`int`, *optional*, defaults to 3):
            Argument cuda_graph_warmup_steps. Number of CUDA graph warmup steps
        external_cuda_graph (`bool`, *optional*, defaults to `False`):
            Argument external_cuda_graph. Use CUDA graph capture and replay. The CUDA graphs aremanually captured
            in the training script.
        cuda_graph_scope (`str`, *optional*, defaults to "full"):
            Argument cuda_graph_scope. Determines the CUDA graphs capturing scope. Valid values are "full" and
            "attn". "Full" scope captures a whole Transformer layer. "Attn" scope only captures operations
            in TransformerLayer._forward_attention().
        inference_max_batch_size (`int`, *optional*, defaults to 8):
            Argument inference_max_batch_size. Maximum number of requests for inference.
        inference_max_seq_length (`int`, *optional*, defaults to 2560):
            Argument inference_max_seq_length. Maximum sequence length expected for inference (prefill + decode).
        inference_dynamic_batching (`bool`, *optional*, defaults to `False`):
            Argument inference_dynamic_batching. Enable dynamic batching mode.
        inference_dynamic_batching_buffer_size_gb (`float`, *optional*, defaults to 40.0):
            Argument inference_dynamic_batching_buffer_size_gb. Total buffer size (GB) allocated for the chunked
            KV memory.
        inference_dynamic_batching_chunk_size (`int`, *optional*, defaults to 256):
            Argument inference_dynamic_batching_chunk_size. KV cache chunk size. It should be a multiple of 256
        inference_dynamic_batching_buffer_guaranteed_fraction (`float`, *optional*, defaults to 0.2):
            Argument inference_dynamic_batching_buffer_guaranteed_fraction. Space is reserved within the
            inference context memory buffer to guarantee that a minimum number of active requests will always be
            able to run to completion. This is to avoid the context being blocked by paused requests.
        inference_dynamic_batching_buffer_overflow_factor (`float`):
            Argument inference_dynamic_batching_buffer_overflow_factor. Scaling factor over the memory buffer size
            for auto computing `max_requests` and `max_tokens`. This scaling factor is used for fitting more
            requests and tokens in the memory buffer than it can safely hold, which in turn increases throughput.
        inference_dynamic_batching_max_requests_override (`int`):
            Argument inference_dynamic_batching_max_requests_override. If set, this overrides the max requests
            as computed from `--inference-dynamic-batching-buffer-overflow-factor`.
        inference_dynamic_batching_max_tokens_override (`int`):
            Argument inference_dynamic_batching_max_tokens_override. If set, this overrides the max tokens as
            computed from `--inference-dynamic-batching-buffer-overflow-factor`.
        symmetric_ar_type (`str`):
            Argument symmetric_ar_type. What type of symmetric all reduce to use. The default is none which is no
            use of symetric memory
        nccl_all_reduce_for_prefill (`bool`, *optional*, defaults to `False`):
            Argument nccl_all_reduce_for_prefill. When using symmeric all reduce kernels this will use regular
            nccl kernels for prefill. This can be more effecient when prefill is large as the nccl kernels can be
            more bandwith optimized
        mlp_chunks_for_prefill (`int`, *optional*, defaults to 1):
            Argument mlp_chunks_for_prefill. Number of chunks along sequence dimension for MLP computation
            during prefill
        fp8 (`str`):
            Argument fp8. Which fp8 format scheme to use for FP8 tensors in the forward and backward pass
        fp8_recipe (`str`, *optional*, defaults to "delayed"):
            Argument fp8_recipe. Which fp8 recipe to use for FP8 tensors in the forward and backward pass
        fp8_margin (`int`, *optional*, defaults to 0):
            Argument fp8_margin. Scaling margin for fp8
        fp8_interval (`int`, *optional*, defaults to 1):
            Argument fp8_interval. DEPRECATED. This flag is ignored. Scaling update interval for fp8
        fp8_amax_history_len (`int`, *optional*, defaults to 1):
            Argument fp8_amax_history_len. Number of steps for which amax history is recorded per tensor
        fp8_amax_compute_algo (`str`, *optional*, defaults to "most_recent"):
            Argument fp8_amax_compute_algo. Algorithm for computing amax from history
        fp8_wgrad (`bool`, *optional*, defaults to `True`):
            Argument fp8_wgrad. Execute wgrad in higher precision even for FP8 runs
        transformer_impl (`str`, *optional*, defaults to "transformer_engine"):
            Argument transformer_impl. Which Transformer implementation to use.
        fp8_param_gather (`bool`, *optional*, defaults to `False`):
            Argument fp8_param_gather. Keep the compute param in fp8 (do not use any other intermediate dtype)
            and perform the param all-gather in fp8.
        first_last_layers_bf16 (`bool`, *optional*, defaults to `False`):
            Argument first_last_layers_bf16. Construct first and last layers in bf16 when doing FP8 training.
        num_layers_at_start_in_bf16 (`int`, *optional*, defaults to 1):
            Argument num_layers_at_start_in_bf16. Number of layers at start to construct in bf16
            when --first-last-layers-bf16 is enabled.
        num_layers_at_end_in_bf16 (`int`, *optional*, defaults to 1):
            Argument num_layers_at_end_in_bf16. Number of layers at end to construct in bf16
            when --first-last-layers-bf16 is enabled.
        te_rng_tracker (`bool`, *optional*, defaults to `False`):
            Argument te_rng_tracker. Use the Transformer Engine version of the random number generator. Required
            for CUDA graphs support.
        inference_rng_tracker (`bool`, *optional*, defaults to `False`):
            Argument inference_rng_tracker. Use a random number generator configured for inference.
        retro_project_dir (`str`):
            Argument retro_project_dir. Retro project directory, which contains the preprocessed data for
            pretraining. This directory is built during preprocessing (see tools/retro/README.md), and
            contains subdirectories for the chunk database and pretraining neighbors.
        retro_add_retriever (`bool`, *optional*, defaults to `False`):
            Argument retro_add_retriever. Add a retriever to the transformer, for use in pretraining a Retro model.
        retro_cyclic_train_iters (`int`):
            Argument retro_cyclic_train_iters. Set number of training iterations for cyclic Retro training.
        retro_encoder_layers (`int`, *optional*, defaults to 2):
            Argument retro_encoder_layers. Number of layers to use for the retrieval encoder.
        retro_encoder_hidden_dropout (`float`, *optional*, defaults to 0.1):
            Argument retro_encoder_hidden_dropout. Hidden dropout for retrieval encoder.
        retro_encoder_attention_dropout (`float`, *optional*, defaults to 0.1):
            Argument retro_encoder_attention_dropout. Attention dropout for retrieval encoder.
        retro_num_neighbors (`int`, *optional*, defaults to 2):
            Argument retro_num_neighbors. Number of neighbors to retrieve during pretraining.
        retro_num_retrieved_chunks (`int`, *optional*, defaults to 2):
            Argument retro_num_retrieved_chunks. Number of chunks to retrieve from the retrieval database.
        retro_attention_gate (`float`, *optional*, defaults to 1):
            Argument retro_attention_gate. Gated cross attention.
        retro_verify_neighbor_count (`bool`, *optional*, defaults to `True`):
            Argument retro_verify_neighbor_count. Skip verifying that len(GPT dataset) == len(saved neighbors).
        enable_experimental (`bool`, *optional*, defaults to `False`):
            Argument enable_experimental. Enable experimental features.
        spec (`list[str]`):
            Argument spec. Specify the <module_location function_name> pair that returns a spec to customize a
            model, transformer block, or transformer layer, depending on the use case.To use local spec specify
            local as the argument.For more details, see the model class, `transformer_block.py`,
            or `transformer_layer.py`
        hybrid_attention_ratio (`float`, *optional*, defaults to 0.0):
            Argument hybrid_attention_ratio. Ratio of attention layers to total layers, in the range [0.0, 1.0].
        hybrid_mlp_ratio (`float`, *optional*, defaults to 0.0):
            Argument hybrid_mlp_ratio. Ratio of mlp layers to total layers, in the range [0.0, 1.0].
        hybrid_override_pattern (`str`):
            Argument hybrid_override_pattern. Force a specific hybrid layer pattern. The valueshould be a string
            of characters chosen fromcore.ssm.mamba_hybrid_layer_allocation.Symbols.If a value greater than 0.0
            is supplied to any of the hybrid ratio arguments, then the number of each typeof layer in the
            override pattern must match number inthe overidden pattern
        mamba_state_dim (`int`, *optional*, defaults to 128):
            Argument mamba_state_dim. State dimension for Mamba layers.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Argument mamba_head_dim. Head dimension for Mamba layers.
        mamba_num_groups (`int`, *optional*, defaults to 8):
            Argument mamba_num_groups. Number of groups for Mamba layers.
        mamba_num_heads (`int`):
            Argument mamba_num_heads. Number of heads for Mamba layers.If not set, then the number of heads will
            be --hidden-size * expand // --mamba-head-dim
        is_hybrid_model (`bool`, *optional*, defaults to `False`):
            Argument is_hybrid_model. Indicates whether the model is a hybrid model.
        disable_mamba_mem_eff_path (`bool`, *optional*, defaults to `False`):
            Argument disable_mamba_mem_eff_path. Disable Mamba efficient path.
        yaml_cfg (`str`):
            Argument yaml_cfg. Config file to add additional arguments
        use_precision_aware_optimizer (`bool`, *optional*, defaults to `False`):
            Argument use_precision_aware_optimizer. Use the precision-aware optimizer in TransformerEngine,
            which allows setting the main params and optimizer states to lower precision, such as fp16, bf16 and fp8.
        main_grads_dtype (`str`, *optional*, defaults to "fp32"):
            Argument main_grads_dtype. Dtype of main grads when enabling precision-aware-optimizer
        main_params_dtype (`str`, *optional*, defaults to "fp32"):
            Argument main_params_dtype. Dtype of main params when enabling precision-aware-optimizer
        exp_avg_dtype (`str`, *optional*, defaults to "fp32"):
            Argument exp_avg_dtype. Dtype of exp_avg (1st moment in adam optimizer) when
            enabling precision-aware-optimizer. This dtype is used for storing the optimizer state in memory
            during training but does not affect the precision in the kernel computation.
        exp_avg_sq_dtype (`str`, *optional*, defaults to "fp32"):
            Argument exp_avg_sq_dtype. Dtype of exp_avg_sq (2nd moment in adam optimizer) when
            enabling precision-aware-optimizer. This dtype is used for storing the optimizer state in memory
            during training but does not affect the precision in the kernel computation.
        enable_one_logger (`bool`, *optional*, defaults to `True`):
            Argument enable_one_logger. If set, disable using one_logger to track E2E metricsNote that one_logger is
            an internal tool and not available externally. For installation, please go
            to https://confluence.nvidia.com/display/MLWFO/Package+Repositoriesfor more details
        one_logger_project (`str`, *optional*, defaults to "megatron-lm"):
            Argument one_logger_project. The one-logger project name. Will ignore if --no-one-logger is set
        one_logger_run_name (`str`):
            Argument one_logger_run_name. The one-logger run name displayed. Will ignore if --no-one-logger is set
        one_logger_async (`bool`, *optional*, defaults to `False`):
            Argument one_logger_async. If set, forces one_logger to use async mode.
        app_tag_run_name (`str`):
            Argument app_tag_run_name. Jobs belonging to same training run, suppose to have the same name. It will
            be used to track progress of a training done over multiple different jobs
        app_tag_run_version (`str`, *optional*, defaults to "0.0.0"):
            Argument app_tag_run_version. The version of the training of which current job is part of. It will be
            used to track the changes in the application side which might change the performance baseline
        inprocess_restart (`bool`, *optional*, defaults to `False`):
            Argument inprocess_restart. Enables in-process restart.
        inprocess_max_iterations (`int`):
            Argument inprocess_max_iterations. Maximum number of in-process restart iterations.
        inprocess_monitor_thread_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_monitor_thread_interval. Monitoring interval (in seconds) for the monitoring thread.
        inprocess_monitor_process_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_monitor_process_interval. Monitoring interval (in seconds) for the monitoring process.
        inprocess_progress_watchdog_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_progress_watchdog_interval. Interval (in seconds) for automatic progress
            watchdog timestamp updates.
        inprocess_heartbeat_interval (`float`, *optional*, defaults to 30):
            Argument inprocess_heartbeat_interval. Monitoring interval (in seconds) for detecting unresponsive ranks.
        inprocess_soft_timeout (`float`, *optional*, defaults to 60):
            Argument inprocess_soft_timeout. Soft progress timeout (in seconds).
        inprocess_hard_timeout (`float`, *optional*, defaults to 90):
            Argument inprocess_hard_timeout. Hard progress timeout (in seconds).
        inprocess_heartbeat_timeout (`float`, *optional*, defaults to 60):
            Argument inprocess_heartbeat_timeout. Timeout (in seconds) for a missing rank detection heartbeat.
        inprocess_barrier_timeout (`float`, *optional*, defaults to 120):
            Argument inprocess_barrier_timeout. Timeout (in seconds) for internal distributed barrier
        inprocess_completion_timeout (`float`, *optional*, defaults to 120):
            Argument inprocess_completion_timeout. Timeout (in seconds) for barrier on completion on all ranks
        inprocess_last_call_wait (`float`, *optional*, defaults to 1):
            Argument inprocess_last_call_wait. Time interval (in seconds) for other ranks to report concurrent
            terminal failures.
        inprocess_termination_grace_time (`float`, *optional*, defaults to 1):
            Argument inprocess_termination_grace_time. Interval (in seconds) between SIGTERM and SIGKILL issued on
            hard timeout
        inprocess_granularity (`str`, *optional*, defaults to "node"):
            Argument inprocess_granularity. Granularity for in-process restart.
        inprocess_active_world_size (`int`, *optional*, defaults to 1):
            Argument inprocess_active_world_size. The number of ranks initially executing the workload. The
            remaining ranks from the allocation are set aside as warm reserve.
        inprocess_empty_cuda_cache (`bool`, *optional*, defaults to `False`):
            Argument inprocess_empty_cuda_cache. Release all unoccupied cached GPU memory on every in-process restart.
        enable_ft_package (`bool`, *optional*, defaults to `False`):
            Argument enable_ft_package. If set, Fault Tolerance package is enabled. Note: This feature is for
            Nvidia internal use only.
        calc_ft_timeouts (`bool`, *optional*, defaults to `False`):
            Argument calc_ft_timeouts. If set, FT package will try to automatically compute the timeouts. Note:
            This feature is for Nvidia internal use only.
        config_logger_dir (`str`, *optional*, defaults to ""):
            Argument config_logger_dir. If set, will dump all configs to --config-logger-dir
        error_injection_rate (`int`, *optional*, defaults to 0):
            Argument error_injection_rate. Rate at which to inject unexpected results, e.g. 1000 means once every
            1000 result validations
        error_injection_type (`str`, *optional*, defaults to "transient_error"):
            Argument error_injection_type. Type of error to inject.
        rerun_mode (`str`, *optional*, defaults to "disabled"):
            Argument rerun_mode. Use re-run engine to validate results (default) or to emit stats on variability
            of computations due to non-deterministic algorithms.
        enable_msc (`bool`, *optional*, defaults to `True`):
            Argument enable_msc. Disable the usage of Multi-Storage Client (MSC) in Megatron Core.
        kitchen_config_file (`str`):
            Argument kitchen_config_file. Use the config .yaml file at the specified location to configure
            kitchen quantization.
        kitchen_recipe_number (`int`):
            Argument kitchen_recipe_number. Use a default kitchen recipe for all layers as defined by QAT_PARAMS index
        sft (`bool`, *optional*, defaults to `False`):
            Argument sft. Megatron SFT training
        sft_tokenizer_prompt_format (`str`, *optional*, defaults to "nemotron-h-aligned"):
            Argument sft_tokenizer_prompt_format. SFT prompt format.
        pad_token_id (`int`, *optional*, defaults to 0):
            Argument pad_token_id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Argument bos_token_id.
        eos_token_id (`int`, *optional*, defaults to 0):
            Argument eos_token_id.
        use_cache (`bool`, *optional*, defaults to `True`):
            Argument use_cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Argument tie_word_embeddings.
    """

    def __init__(
        self,
        num_layers: int = None,
        encoder_num_layers: int = None,
        decoder_num_layers: int = None,
        hidden_size: int = None,
        ffn_hidden_size: int = None,
        num_attention_heads: int = None,
        attention_backend: str = "default",
        kv_channels: int = None,
        group_query_attention: bool = False,
        num_query_groups: int = 1,
        max_position_embeddings: int = None,
        position_embedding_type: str = "learned_absolute",
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        use_rotary_position_embeddings: bool = False,
        rotary_base: int = 10000,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        rotary_seq_len_interpolation_factor: int = None,
        use_rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        no_rope_freq: Any = None,
        add_position_embedding: bool = True,
        mrope_section: list[int] = None,
        make_vocab_size_divisible_by: int = 128,
        normalization: None = "LayerNorm",
        norm_epsilon: float = 1e-05,
        apply_layernorm_1p: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        openai_gelu: bool = False,
        squared_relu: bool = False,
        swiglu: bool = False,
        onnx_safe: bool = None,
        bert_binary_head: bool = True,
        untie_embeddings_and_output_weights: bool = False,
        multi_latent_attention: bool = False,
        mtp_num_layers: int = None,
        mtp_loss_scaling_factor: float = 0.1,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        weight_decay: float = 0.01,
        start_weight_decay: float = None,
        end_weight_decay: float = None,
        weight_decay_incr_style: str = "constant",
        clip_grad: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-08,
        sgd_momentum: float = 0.9,
        micro_batch_size: int = None,
        batch_size: int = None,
        global_batch_size: int = None,
        rampup_batch_size: list[str] = None,
        decrease_batch_size_if_needed: bool = False,
        recompute_activations: bool = False,
        recompute_granularity: str = None,
        check_for_nan_in_loss_and_grad: bool = True,
        check_for_spiky_loss: bool = False,
        check_for_large_grads: bool = False,
        distribute_saved_activations: bool = False,
        recompute_method: str = None,
        recompute_num_layers: int = None,
        recompute_modules: list[str] = None,
        clone_scatter_output_in_embedding: bool = True,
        profile: bool = False,
        profile_step_start: int = 10,
        profile_step_end: int = 12,
        iterations_to_skip: list[int] = [],
        result_rejected_tracker_filename: str = None,
        enable_gloo_process_groups: bool = True,
        use_pytorch_profiler: bool = False,
        profile_ranks: list[int] = [0],
        record_memory_history: bool = False,
        memory_snapshot_path: str = "snapshot.pickle",
        tp_comm_overlap: bool = False,
        tp_comm_overlap_cfg: str = None,
        tp_comm_overlap_ag: bool = True,
        tp_comm_overlap_rs: bool = True,
        tp_comm_overlap_rs_dgrad: bool = False,
        tp_comm_bulk_dgrad: bool = True,
        tp_comm_bulk_wgrad: bool = True,
        tp_comm_bootstrap_backend: str = "nccl",
        use_cpu_initialization: bool = None,
        empty_unused_memory_level: int = 0,
        deterministic_mode: bool = False,
        check_weight_hash_across_dp_replicas_interval: int = None,
        calculate_per_token_loss: bool = False,
        train_sync_interval: int = None,
        checkpoint_activations: bool = False,
        train_iters: int = None,
        train_samples: int = None,
        log_interval: int = 100,
        exit_interval: int = None,
        exit_duration_in_mins: int = None,
        exit_signal_handler: bool = False,
        tensorboard_dir: str = None,
        masked_softmax_fusion: bool = True,
        bias_gelu_fusion: bool = True,
        bias_swiglu_fusion: bool = True,
        bias_dropout_fusion: bool = True,
        apply_rope_fusion: bool = True,
        cross_entropy_loss_fusion: bool = False,
        cross_entropy_fusion_impl: str = "native",
        use_flash_attn: bool = False,
        add_bias_linear: bool = True,
        add_qkv_bias: bool = False,
        optimizer: str = "adam",
        optimizer_cpu_offload: bool = False,
        optimizer_offload_fraction: float = 1.0,
        use_torch_optimizer_for_cpu_offload: bool = False,
        overlap_cpu_optimizer_d2h_h2d: bool = False,
        pin_cpu_grads: bool = True,
        pin_cpu_params: bool = True,
        dataloader_type: str = None,
        async_tensor_model_parallel_allreduce: bool = True,
        no_persist_layer_norm: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = True,
        deprecated_use_mcore_models: bool = False,
        use_legacy_models: bool = False,
        manual_gc: bool = False,
        manual_gc_interval: int = 0,
        manual_gc_eval: bool = True,
        tp_comm_split_ag: bool = True,
        tp_comm_split_rs: bool = True,
        pipeline_model_parallel_comm_backend: str = None,
        high_priority_stream_groups: list[str] = [],
        seed: int = 1234,
        data_parallel_random_init: bool = False,
        init_method_std: float = 0.02,
        init_method_xavier_uniform: bool = False,
        lr: float = None,
        lr_decay_style: str = "linear",
        lr_wsd_decay_style: str = "exponential",
        lr_decay_iters: int = None,
        lr_decay_samples: int = None,
        lr_wsd_decay_samples: int = None,
        lr_wsd_decay_iters: int = None,
        lr_warmup_fraction: float = None,
        lr_warmup_iters: int = 0,
        lr_warmup_samples: int = 0,
        lr_warmup_init: float = 0.0,
        warmup: int = None,
        min_lr: float = 0.0,
        override_opt_param_scheduler: bool = False,
        use_checkpoint_opt_param_scheduler: bool = False,
        decoupled_lr: float = None,
        decoupled_min_lr: float = None,
        save: str = None,
        save_interval: int = None,
        no_save_optim: bool = None,
        no_save_rng: bool = None,
        load: str = None,
        no_load_optim: bool = None,
        no_load_rng: bool = None,
        non_persistent_save_interval: int = None,
        non_persistent_ckpt_type: str = None,
        non_persistent_global_ckpt_dir: str = None,
        non_persistent_local_ckpt_dir: str = None,
        non_persistent_local_ckpt_algo: str = "fully_parallel",
        finetune: bool = False,
        pretrained_checkpoint: str = None,
        ckpt_step: int = None,
        perform_initialization: bool = True,
        use_checkpoint_args: bool = False,
        use_mp_args_from_checkpoint_args: bool = False,
        use_tokenizer_model_from_checkpoint_args: bool = True,
        exit_on_missing_checkpoint: bool = False,
        use_dist_ckpt_deprecated: bool = False,
        use_persistent_ckpt_worker: bool = False,
        auto_detect_ckpt_format: bool = False,
        dist_ckpt_format_deprecated: None = None,
        ckpt_format: None = "torch_dist",
        ckpt_convert_format: None = None,
        ckpt_convert_save: None = None,
        ckpt_convert_update_legacy_dist_opt_format: bool = False,
        ckpt_fully_parallel_save_deprecated: bool = False,
        ckpt_fully_parallel_save: bool = True,
        async_save: bool = None,
        ckpt_fully_parallel_load: bool = False,
        ckpt_assume_constant_structure: bool = False,
        dist_ckpt_strictness: str = "assume_ok_unexpected",
        load_model_opt_format: bool = False,
        fp16: bool = False,
        bf16: bool = False,
        grad_reduce_in_bf16: bool = False,
        loss_scale: float = None,
        initial_loss_scale: float = 4294967296,
        min_loss_scale: float = 1.0,
        loss_scale_window: float = 1000,
        hysteresis: int = 2,
        fp32_residual_connection: bool = False,
        apply_query_key_layer_scaling: bool = False,
        attention_softmax_in_fp32: bool = False,
        accumulate_allreduce_grads_in_fp32: bool = False,
        fp16_lm_cross_entropy: bool = False,
        disable_bf16_reduced_precision_matmul: bool = False,
        reuse_grad_buf_for_mxfp8_param_ag: bool = False,
        tensor_model_parallel_size: int = 1,
        encoder_tensor_model_parallel_size: int = 0,
        pipeline_model_parallel_size: int = 1,
        encoder_pipeline_model_parallel_size: int = 0,
        pipeline_model_parallel_split_rank: int = None,
        decoder_first_pipeline_num_layers: int = None,
        decoder_last_pipeline_num_layers: int = None,
        pipeline_model_parallel_layout: str = None,
        model_parallel_size: int = None,
        num_layers_per_virtual_pipeline_stage: int = None,
        num_virtual_stages_per_pipeline_rank: int = None,
        microbatch_group_size_per_vp_stage: int = None,
        overlap_p2p_comm: bool = True,
        overlap_p2p_comm_warmup_flush: bool = False,
        distributed_backend: None = "nccl",
        distributed_timeout_minutes: int = 10,
        overlap_grad_reduce: bool = False,
        defer_embedding_wgrad_compute: bool = False,
        wgrad_deferral_limit: int = 0,
        align_grad_reduce: bool = True,
        ddp_num_buckets: int = None,
        ddp_bucket_size: int = None,
        ddp_pad_buckets_for_high_nccl_busbw: bool = False,
        ddp_average_in_collective: bool = False,
        overlap_param_gather: bool = False,
        overlap_param_gather_with_optimizer_step: bool = False,
        align_param_gather: bool = True,
        scatter_gather_tensors_in_pipeline: bool = True,
        use_ring_exchange_p2p: bool = False,
        local_rank: int = 0,
        lazy_mpu_init: bool = None,
        account_for_embedding_in_pipeline_split: bool = False,
        account_for_loss_in_pipeline_split: bool = False,
        use_distributed_optimizer: bool = False,
        nccl_ub: bool = False,
        use_sharp: bool = False,
        use_custom_fsdp: bool = False,
        init_model_with_meta_device: bool = False,
        data_parallel_sharding_strategy: str = "no_shard",
        gradient_reduce_div_fusion: bool = True,
        fsdp_double_buffer: bool = False,
        suggested_communication_unit_size: int = None,
        keep_fp8_transpose_cache_when_using_custom_fsdp: bool = False,
        num_distributed_optimizer_instances: int = 1,
        use_torch_fsdp2: bool = False,
        torch_fsdp2_reshard_after_forward: bool = True,
        context_parallel_size: int = 1,
        cp_comm_type: list[str] = ["p2p"],
        hierarchical_context_parallel_sizes: list[int] = None,
        nccl_communicator_config_path: str = None,
        use_tp_pp_dp_mapping: bool = False,
        replication: bool = False,
        replication_jump: int = None,
        replication_factor: int = 2,
        eval_iters: int = 100,
        eval_interval: int = 1000,
        test_mode: bool = False,
        skip_train: bool = False,
        data_path: list[str] = None,
        split: str = None,
        train_data_path: list[str] = None,
        valid_data_path: list[str] = None,
        test_data_path: list[str] = None,
        data_args_path: str = None,
        per_split_data_args_path: str = None,
        data_cache_path: None = None,
        mmap_bin_files: bool = True,
        mock_data: bool = False,
        seq_length: int = None,
        encoder_seq_length: int = None,
        decoder_seq_length: int = None,
        retriever_seq_length: int = 256,
        sample_rate: float = 1.0,
        mask_prob: float = 0.15,
        short_seq_prob: float = 0.1,
        num_workers: int = 2,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        create_attention_mask_in_dataloader: bool = True,
        num_dataset_builder_threads: int = 1,
        object_storage_cache_path: str = None,
        mid_level_dataset_surplus: float = 0.005,
        vocab_size: int = None,
        vocab_file: str = None,
        merge_file: str = None,
        vocab_extra_ids: int = 0,
        tokenizer_type: str = None,
        tokenizer_model: str = None,
        tiktoken_pattern: str = None,
        tiktoken_num_special_tokens: int = 1000,
        tiktoken_special_tokens: list[str] = None,
        adlr_autoresume: bool = False,
        adlr_autoresume_interval: int = 1000,
        ict_head_size: int = None,
        biencoder_projection_dim: int = 0,
        biencoder_shared_query_context_model: bool = False,
        ict_load: str = None,
        bert_load: str = None,
        titles_data_path: str = None,
        query_in_block_prob: float = 0.1,
        use_one_sent_docs: bool = False,
        evidence_data_path: str = None,
        retriever_report_topk_accuracies: list[int] = [],
        retriever_score_scaling: bool = False,
        block_data_path: str = None,
        embedding_path: str = None,
        indexer_batch_size: int = 128,
        indexer_log_interval: int = 1000,
        num_classes: int = 1000,
        img_h: int = 224,
        img_w: int = 224,
        num_channels: int = 3,
        patch_dim: int = 16,
        classes_fraction: float = 1.0,
        data_per_class_fraction: float = 1.0,
        data_sharding: bool = True,
        head_lr_mult: float = 1.0,
        vision_pretraining: bool = False,
        vision_pretraining_type: str = "classify",
        vision_backbone_type: str = "vit",
        swin_backbone_type: str = "tiny",
        mask_type: str = "random",
        mask_factor: float = 1.0,
        iter_per_epoch: int = 1250,
        dino_local_img_size: int = 96,
        dino_local_crops_number: int = 10,
        dino_head_hidden_size: int = 2048,
        dino_bottleneck_size: int = 256,
        dino_freeze_last_layer: float = 1,
        dino_norm_last_layer: bool = False,
        dino_warmup_teacher_temp: float = 0.04,
        dino_teacher_temp: float = 0.07,
        dino_warmup_teacher_temp_epochs: int = 30,
        qk_layernorm: bool = False,
        qk_l2_norm: bool = False,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = None,
        num_experts: int = None,
        moe_layer_freq: Any = 1,
        moe_ffn_hidden_size: int = None,
        moe_shared_expert_intermediate_size: int = None,
        moe_shared_expert_overlap: bool = False,
        moe_grouped_gemm: bool = False,
        moe_use_legacy_grouped_gemm: bool = False,
        moe_layer_recompute: bool = False,
        moe_extended_tp: bool = False,
        moe_use_upcycling: bool = False,
        moe_router_load_balancing_type: str = "aux_loss",
        moe_router_dtype: str = None,
        moe_router_score_function: str = "softmax",
        moe_router_topk: int = 2,
        moe_router_pre_softmax: bool = False,
        moe_router_num_groups: int = None,
        moe_router_group_topk: int = None,
        moe_router_topk_scaling_factor: float = None,
        moe_router_enable_expert_bias: bool = False,
        moe_router_bias_update_rate: float = 0.001,
        moe_router_force_load_balancing: bool = False,
        moe_router_padding_for_fp8: bool = False,
        moe_aux_loss_coeff: float = 0.0,
        moe_z_loss_coeff: float = None,
        moe_input_jitter_eps: float = None,
        moe_per_layer_logging: bool = False,
        moe_token_dispatcher_type: str = "allgather",
        moe_enable_deepep: bool = False,
        moe_deepep_num_sms: int = 20,
        moe_permute_fusion: bool = False,
        moe_expert_capacity_factor: float = None,
        moe_pad_expert_input_to_capacity: bool = False,
        moe_token_drop_policy: str = "probs",
        moe_apply_probs_on_input: bool = False,
        delay_wgrad_compute: bool = False,
        moe_upcycling_granularity: int = 1,
        q_lora_rank: int = None,
        kv_lora_rank: int = 32,
        qk_head_dim: int = 128,
        qk_pos_emb_head_dim: int = 64,
        v_head_dim: int = 128,
        rotary_scaling_factor: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
        heterogeneous_layers_config_path: str = None,
        heterogeneous_layers_config_encoded_json: str = None,
        log_params_norm: bool = False,
        log_num_zeros_in_grad: bool = False,
        log_throughput: bool = False,
        log_progress: bool = False,
        timing_log_level: int = 0,
        log_energy: bool = False,
        barrier_with_L1_time: bool = True,
        timing_log_option: str = "minmax",
        tensorboard_log_interval: int = 1,
        tensorboard_queue_size: int = 1000,
        log_timers_to_tensorboard: bool = False,
        log_loss_scale_to_tensorboard: bool = True,
        log_validation_ppl_to_tensorboard: bool = False,
        log_memory_to_tensorboard: bool = False,
        log_world_size_to_tensorboard: bool = False,
        wandb_project: str = "",
        wandb_exp_name: str = "",
        wandb_save_dir: str = "",
        logging_level: int = None,
        log_straggler: bool = False,
        disable_straggler_on_startup: bool = False,
        straggler_ctrlr_port: int = 65535,
        straggler_minmax_count: int = 1,
        run_workload_inspector_server: bool = False,
        inference_batch_times_seqlen_threshold: int = -1,
        max_tokens_to_oom: int = 12000,
        output_bert_embeddings: bool = False,
        bert_embedder_type: None = "megatron",
        flash_decode: bool = False,
        enable_cuda_graph: bool = False,
        cuda_graph_warmup_steps: int = 3,
        external_cuda_graph: bool = False,
        cuda_graph_scope: str = "full",
        inference_max_batch_size: int = 8,
        inference_max_seq_length: int = 2560,
        inference_dynamic_batching: bool = False,
        inference_dynamic_batching_buffer_size_gb: float = 40.0,
        inference_dynamic_batching_chunk_size: int = 256,
        inference_dynamic_batching_buffer_guaranteed_fraction: float = 0.2,
        inference_dynamic_batching_buffer_overflow_factor: float = None,
        inference_dynamic_batching_max_requests_override: int = None,
        inference_dynamic_batching_max_tokens_override: int = None,
        symmetric_ar_type: str = None,
        nccl_all_reduce_for_prefill: bool = False,
        mlp_chunks_for_prefill: int = 1,
        fp8: None = None,
        fp8_recipe: None = "delayed",
        fp8_margin: int = 0,
        fp8_interval: int = 1,
        fp8_amax_history_len: int = 1,
        fp8_amax_compute_algo: None = "most_recent",
        fp8_wgrad: bool = True,
        transformer_impl: None = "transformer_engine",
        fp8_param_gather: bool = False,
        first_last_layers_bf16: bool = False,
        num_layers_at_start_in_bf16: int = 1,
        num_layers_at_end_in_bf16: int = 1,
        te_rng_tracker: bool = False,
        inference_rng_tracker: bool = False,
        retro_project_dir: None = None,
        retro_add_retriever: bool = False,
        retro_cyclic_train_iters: int = None,
        retro_encoder_layers: int = 2,
        retro_encoder_hidden_dropout: float = 0.1,
        retro_encoder_attention_dropout: float = 0.1,
        retro_num_neighbors: int = 2,
        retro_num_retrieved_chunks: int = 2,
        retro_attention_gate: float = 1,
        retro_verify_neighbor_count: bool = True,
        enable_experimental: bool = False,
        spec: list[str] = None,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        mamba_state_dim: int = 128,
        mamba_head_dim: int = 64,
        mamba_num_groups: int = 8,
        mamba_num_heads: int = None,
        is_hybrid_model: bool = False,
        disable_mamba_mem_eff_path: bool = False,
        yaml_cfg: str = None,
        use_precision_aware_optimizer: bool = False,
        main_grads_dtype: None = "fp32",
        main_params_dtype: None = "fp32",
        exp_avg_dtype: None = "fp32",
        exp_avg_sq_dtype: None = "fp32",
        enable_one_logger: bool = True,
        one_logger_project: str = "megatron-lm",
        one_logger_run_name: str = None,
        one_logger_async: bool = False,
        app_tag_run_name: str = None,
        app_tag_run_version: str = "0.0.0",
        inprocess_restart: bool = False,
        inprocess_max_iterations: int = None,
        inprocess_monitor_thread_interval: float = 1.0,
        inprocess_monitor_process_interval: float = 1.0,
        inprocess_progress_watchdog_interval: float = 1.0,
        inprocess_heartbeat_interval: float = 30,
        inprocess_soft_timeout: float = 60,
        inprocess_hard_timeout: float = 90,
        inprocess_heartbeat_timeout: float = 60,
        inprocess_barrier_timeout: float = 120,
        inprocess_completion_timeout: float = 120,
        inprocess_last_call_wait: float = 1,
        inprocess_termination_grace_time: float = 1,
        inprocess_granularity: str = "node",
        inprocess_active_world_size: int = 1,
        inprocess_empty_cuda_cache: bool = False,
        enable_ft_package: bool = False,
        calc_ft_timeouts: bool = False,
        config_logger_dir: str = "",
        error_injection_rate: int = 0,
        error_injection_type: str = "transient_error",
        rerun_mode: str = "disabled",
        enable_msc: bool = True,
        kitchen_config_file: str = None,
        kitchen_recipe_number: int = None,
        sft: bool = False,
        sft_tokenizer_prompt_format: str = "nemotron-h-aligned",
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.num_layers: int = num_layers
        self.encoder_num_layers: int = encoder_num_layers
        self.decoder_num_layers: int = decoder_num_layers
        self.hidden_size: int = hidden_size
        self.ffn_hidden_size: int = ffn_hidden_size
        self.num_attention_heads: int = num_attention_heads
        self.attention_backend: str = attention_backend
        self.kv_channels: int = kv_channels
        self.group_query_attention: bool = group_query_attention
        self.num_query_groups: int = num_query_groups
        self.max_position_embeddings: int = max_position_embeddings
        self.position_embedding_type: str = position_embedding_type
        self.relative_attention_num_buckets: int = relative_attention_num_buckets
        self.relative_attention_max_distance: int = relative_attention_max_distance
        self.use_rotary_position_embeddings: bool = use_rotary_position_embeddings
        self.rotary_base: int = rotary_base
        self.rotary_percent: float = rotary_percent
        self.rotary_interleaved: bool = rotary_interleaved
        self.rotary_seq_len_interpolation_factor: int = (
            rotary_seq_len_interpolation_factor
        )
        self.use_rope_scaling: bool = use_rope_scaling
        self.rope_scaling_factor: float = rope_scaling_factor
        self.no_rope_freq: Any = no_rope_freq
        self.add_position_embedding: bool = add_position_embedding
        self.mrope_section: list[int] = mrope_section
        self.make_vocab_size_divisible_by: int = make_vocab_size_divisible_by
        self.normalization: None = normalization
        self.norm_epsilon: float = norm_epsilon
        self.apply_layernorm_1p: bool = apply_layernorm_1p
        self.apply_residual_connection_post_layernorm: bool = (
            apply_residual_connection_post_layernorm
        )
        self.openai_gelu: bool = openai_gelu
        self.squared_relu: bool = squared_relu
        self.swiglu: bool = swiglu
        self.onnx_safe: bool = onnx_safe
        self.bert_binary_head: bool = bert_binary_head
        self.untie_embeddings_and_output_weights: bool = (
            untie_embeddings_and_output_weights
        )
        self.multi_latent_attention: bool = multi_latent_attention
        self.mtp_num_layers: int = mtp_num_layers
        self.mtp_loss_scaling_factor: float = mtp_loss_scaling_factor
        self.attention_dropout: float = attention_dropout
        self.hidden_dropout: float = hidden_dropout
        self.weight_decay: float = weight_decay
        self.start_weight_decay: float = start_weight_decay
        self.end_weight_decay: float = end_weight_decay
        self.weight_decay_incr_style: str = weight_decay_incr_style
        self.clip_grad: float = clip_grad
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_eps: float = adam_eps
        self.sgd_momentum: float = sgd_momentum
        self.micro_batch_size: int = micro_batch_size
        self.batch_size: int = batch_size
        self.global_batch_size: int = global_batch_size
        self.rampup_batch_size: list[str] = rampup_batch_size
        self.decrease_batch_size_if_needed: bool = decrease_batch_size_if_needed
        self.recompute_activations: bool = recompute_activations
        self.recompute_granularity: str = recompute_granularity
        self.check_for_nan_in_loss_and_grad: bool = check_for_nan_in_loss_and_grad
        self.check_for_spiky_loss: bool = check_for_spiky_loss
        self.check_for_large_grads: bool = check_for_large_grads
        self.distribute_saved_activations: bool = distribute_saved_activations
        self.recompute_method: str = recompute_method
        self.recompute_num_layers: int = recompute_num_layers
        self.recompute_modules: list[str] = recompute_modules
        self.clone_scatter_output_in_embedding: bool = clone_scatter_output_in_embedding
        self.profile: bool = profile
        self.profile_step_start: int = profile_step_start
        self.profile_step_end: int = profile_step_end
        self.iterations_to_skip: list[int] = iterations_to_skip
        self.result_rejected_tracker_filename: str = result_rejected_tracker_filename
        self.enable_gloo_process_groups: bool = enable_gloo_process_groups
        self.use_pytorch_profiler: bool = use_pytorch_profiler
        self.profile_ranks: list[int] = profile_ranks
        self.record_memory_history: bool = record_memory_history
        self.memory_snapshot_path: str = memory_snapshot_path
        self.tp_comm_overlap: bool = tp_comm_overlap
        self.tp_comm_overlap_cfg: str = tp_comm_overlap_cfg
        self.tp_comm_overlap_ag: bool = tp_comm_overlap_ag
        self.tp_comm_overlap_rs: bool = tp_comm_overlap_rs
        self.tp_comm_overlap_rs_dgrad: bool = tp_comm_overlap_rs_dgrad
        self.tp_comm_bulk_dgrad: bool = tp_comm_bulk_dgrad
        self.tp_comm_bulk_wgrad: bool = tp_comm_bulk_wgrad
        self.tp_comm_bootstrap_backend: str = tp_comm_bootstrap_backend
        self.use_cpu_initialization: bool = use_cpu_initialization
        self.empty_unused_memory_level: int = empty_unused_memory_level
        self.deterministic_mode: bool = deterministic_mode
        self.check_weight_hash_across_dp_replicas_interval: int = (
            check_weight_hash_across_dp_replicas_interval
        )
        self.calculate_per_token_loss: bool = calculate_per_token_loss
        self.train_sync_interval: int = train_sync_interval
        self.checkpoint_activations: bool = checkpoint_activations
        self.train_iters: int = train_iters
        self.train_samples: int = train_samples
        self.log_interval: int = log_interval
        self.exit_interval: int = exit_interval
        self.exit_duration_in_mins: int = exit_duration_in_mins
        self.exit_signal_handler: bool = exit_signal_handler
        self.tensorboard_dir: str = tensorboard_dir
        self.masked_softmax_fusion: bool = masked_softmax_fusion
        self.bias_gelu_fusion: bool = bias_gelu_fusion
        self.bias_swiglu_fusion: bool = bias_swiglu_fusion
        self.bias_dropout_fusion: bool = bias_dropout_fusion
        self.apply_rope_fusion: bool = apply_rope_fusion
        self.cross_entropy_loss_fusion: bool = cross_entropy_loss_fusion
        self.cross_entropy_fusion_impl: str = cross_entropy_fusion_impl
        self.use_flash_attn: bool = use_flash_attn
        self.add_bias_linear: bool = add_bias_linear
        self.add_qkv_bias: bool = add_qkv_bias
        self.optimizer: str = optimizer
        self.optimizer_cpu_offload: bool = optimizer_cpu_offload
        self.optimizer_offload_fraction: float = optimizer_offload_fraction
        self.use_torch_optimizer_for_cpu_offload: bool = (
            use_torch_optimizer_for_cpu_offload
        )
        self.overlap_cpu_optimizer_d2h_h2d: bool = overlap_cpu_optimizer_d2h_h2d
        self.pin_cpu_grads: bool = pin_cpu_grads
        self.pin_cpu_params: bool = pin_cpu_params
        self.dataloader_type: str = dataloader_type
        self.async_tensor_model_parallel_allreduce: bool = (
            async_tensor_model_parallel_allreduce
        )
        self.no_persist_layer_norm: bool = no_persist_layer_norm
        self.sequence_parallel: bool = sequence_parallel
        self.gradient_accumulation_fusion: bool = gradient_accumulation_fusion
        self.deprecated_use_mcore_models: bool = deprecated_use_mcore_models
        self.use_legacy_models: bool = use_legacy_models
        self.manual_gc: bool = manual_gc
        self.manual_gc_interval: int = manual_gc_interval
        self.manual_gc_eval: bool = manual_gc_eval
        self.tp_comm_split_ag: bool = tp_comm_split_ag
        self.tp_comm_split_rs: bool = tp_comm_split_rs
        self.pipeline_model_parallel_comm_backend: str = (
            pipeline_model_parallel_comm_backend
        )
        self.high_priority_stream_groups: list[str] = high_priority_stream_groups
        self.seed: int = seed
        self.data_parallel_random_init: bool = data_parallel_random_init
        self.init_method_std: float = init_method_std
        self.init_method_xavier_uniform: bool = init_method_xavier_uniform
        self.lr: float = lr
        self.lr_decay_style: str = lr_decay_style
        self.lr_wsd_decay_style: str = lr_wsd_decay_style
        self.lr_decay_iters: int = lr_decay_iters
        self.lr_decay_samples: int = lr_decay_samples
        self.lr_wsd_decay_samples: int = lr_wsd_decay_samples
        self.lr_wsd_decay_iters: int = lr_wsd_decay_iters
        self.lr_warmup_fraction: float = lr_warmup_fraction
        self.lr_warmup_iters: int = lr_warmup_iters
        self.lr_warmup_samples: int = lr_warmup_samples
        self.lr_warmup_init: float = lr_warmup_init
        self.warmup: int = warmup
        self.min_lr: float = min_lr
        self.override_opt_param_scheduler: bool = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler: bool = (
            use_checkpoint_opt_param_scheduler
        )
        self.decoupled_lr: float = decoupled_lr
        self.decoupled_min_lr: float = decoupled_min_lr
        self.save: str = save
        self.save_interval: int = save_interval
        self.no_save_optim: bool = no_save_optim
        self.no_save_rng: bool = no_save_rng
        self.load: str = load
        self.no_load_optim: bool = no_load_optim
        self.no_load_rng: bool = no_load_rng
        self.non_persistent_save_interval: int = non_persistent_save_interval
        self.non_persistent_ckpt_type: str = non_persistent_ckpt_type
        self.non_persistent_global_ckpt_dir: str = non_persistent_global_ckpt_dir
        self.non_persistent_local_ckpt_dir: str = non_persistent_local_ckpt_dir
        self.non_persistent_local_ckpt_algo: str = non_persistent_local_ckpt_algo
        self.finetune: bool = finetune
        self.pretrained_checkpoint: str = pretrained_checkpoint
        self.ckpt_step: int = ckpt_step
        self.perform_initialization: bool = perform_initialization
        self.use_checkpoint_args: bool = use_checkpoint_args
        self.use_mp_args_from_checkpoint_args: bool = use_mp_args_from_checkpoint_args
        self.use_tokenizer_model_from_checkpoint_args: bool = (
            use_tokenizer_model_from_checkpoint_args
        )
        self.exit_on_missing_checkpoint: bool = exit_on_missing_checkpoint
        self.use_dist_ckpt_deprecated: bool = use_dist_ckpt_deprecated
        self.use_persistent_ckpt_worker: bool = use_persistent_ckpt_worker
        self.auto_detect_ckpt_format: bool = auto_detect_ckpt_format
        self.dist_ckpt_format_deprecated: None = dist_ckpt_format_deprecated
        self.ckpt_format: None = ckpt_format
        self.ckpt_convert_format: None = ckpt_convert_format
        self.ckpt_convert_save: None = ckpt_convert_save
        self.ckpt_convert_update_legacy_dist_opt_format: bool = (
            ckpt_convert_update_legacy_dist_opt_format
        )
        self.ckpt_fully_parallel_save_deprecated: bool = (
            ckpt_fully_parallel_save_deprecated
        )
        self.ckpt_fully_parallel_save: bool = ckpt_fully_parallel_save
        self.async_save: bool = async_save
        self.ckpt_fully_parallel_load: bool = ckpt_fully_parallel_load
        self.ckpt_assume_constant_structure: bool = ckpt_assume_constant_structure
        self.dist_ckpt_strictness: str = dist_ckpt_strictness
        self.load_model_opt_format: bool = load_model_opt_format
        self.fp16: bool = fp16
        self.bf16: bool = bf16
        self.grad_reduce_in_bf16: bool = grad_reduce_in_bf16
        self.loss_scale: float = loss_scale
        self.initial_loss_scale: float = initial_loss_scale
        self.min_loss_scale: float = min_loss_scale
        self.loss_scale_window: float = loss_scale_window
        self.hysteresis: int = hysteresis
        self.fp32_residual_connection: bool = fp32_residual_connection
        self.apply_query_key_layer_scaling: bool = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32: bool = attention_softmax_in_fp32
        self.accumulate_allreduce_grads_in_fp32: bool = (
            accumulate_allreduce_grads_in_fp32
        )
        self.fp16_lm_cross_entropy: bool = fp16_lm_cross_entropy
        self.disable_bf16_reduced_precision_matmul: bool = (
            disable_bf16_reduced_precision_matmul
        )
        self.reuse_grad_buf_for_mxfp8_param_ag: bool = reuse_grad_buf_for_mxfp8_param_ag
        self.tensor_model_parallel_size: int = tensor_model_parallel_size
        self.encoder_tensor_model_parallel_size: int = (
            encoder_tensor_model_parallel_size
        )
        self.pipeline_model_parallel_size: int = pipeline_model_parallel_size
        self.encoder_pipeline_model_parallel_size: int = (
            encoder_pipeline_model_parallel_size
        )
        self.pipeline_model_parallel_split_rank: int = (
            pipeline_model_parallel_split_rank
        )
        self.decoder_first_pipeline_num_layers: int = decoder_first_pipeline_num_layers
        self.decoder_last_pipeline_num_layers: int = decoder_last_pipeline_num_layers
        self.pipeline_model_parallel_layout: str = pipeline_model_parallel_layout
        self.model_parallel_size: int = model_parallel_size
        self.num_layers_per_virtual_pipeline_stage: int = (
            num_layers_per_virtual_pipeline_stage
        )
        self.num_virtual_stages_per_pipeline_rank: int = (
            num_virtual_stages_per_pipeline_rank
        )
        self.microbatch_group_size_per_vp_stage: int = (
            microbatch_group_size_per_vp_stage
        )
        self.overlap_p2p_comm: bool = overlap_p2p_comm
        self.overlap_p2p_comm_warmup_flush: bool = overlap_p2p_comm_warmup_flush
        self.distributed_backend: None = distributed_backend
        self.distributed_timeout_minutes: int = distributed_timeout_minutes
        self.overlap_grad_reduce: bool = overlap_grad_reduce
        self.defer_embedding_wgrad_compute: bool = defer_embedding_wgrad_compute
        self.wgrad_deferral_limit: int = wgrad_deferral_limit
        self.align_grad_reduce: bool = align_grad_reduce
        self.ddp_num_buckets: int = ddp_num_buckets
        self.ddp_bucket_size: int = ddp_bucket_size
        self.ddp_pad_buckets_for_high_nccl_busbw: bool = (
            ddp_pad_buckets_for_high_nccl_busbw
        )
        self.ddp_average_in_collective: bool = ddp_average_in_collective
        self.overlap_param_gather: bool = overlap_param_gather
        self.overlap_param_gather_with_optimizer_step: bool = (
            overlap_param_gather_with_optimizer_step
        )
        self.align_param_gather: bool = align_param_gather
        self.scatter_gather_tensors_in_pipeline: bool = (
            scatter_gather_tensors_in_pipeline
        )
        self.use_ring_exchange_p2p: bool = use_ring_exchange_p2p
        self.local_rank: int = local_rank
        self.lazy_mpu_init: bool = lazy_mpu_init
        self.account_for_embedding_in_pipeline_split: bool = (
            account_for_embedding_in_pipeline_split
        )
        self.account_for_loss_in_pipeline_split: bool = (
            account_for_loss_in_pipeline_split
        )
        self.use_distributed_optimizer: bool = use_distributed_optimizer
        self.nccl_ub: bool = nccl_ub
        self.use_sharp: bool = use_sharp
        self.use_custom_fsdp: bool = use_custom_fsdp
        self.init_model_with_meta_device: bool = init_model_with_meta_device
        self.data_parallel_sharding_strategy: str = data_parallel_sharding_strategy
        self.gradient_reduce_div_fusion: bool = gradient_reduce_div_fusion
        self.fsdp_double_buffer: bool = fsdp_double_buffer
        self.suggested_communication_unit_size: int = suggested_communication_unit_size
        self.keep_fp8_transpose_cache_when_using_custom_fsdp: bool = (
            keep_fp8_transpose_cache_when_using_custom_fsdp
        )
        self.num_distributed_optimizer_instances: int = (
            num_distributed_optimizer_instances
        )
        self.use_torch_fsdp2: bool = use_torch_fsdp2
        self.torch_fsdp2_reshard_after_forward: bool = torch_fsdp2_reshard_after_forward
        self.context_parallel_size: int = context_parallel_size
        self.cp_comm_type: list[str] = cp_comm_type
        self.hierarchical_context_parallel_sizes: list[int] = (
            hierarchical_context_parallel_sizes
        )
        self.nccl_communicator_config_path: str = nccl_communicator_config_path
        self.use_tp_pp_dp_mapping: bool = use_tp_pp_dp_mapping
        self.replication: bool = replication
        self.replication_jump: int = replication_jump
        self.replication_factor: int = replication_factor
        self.eval_iters: int = eval_iters
        self.eval_interval: int = eval_interval
        self.test_mode: bool = test_mode
        self.skip_train: bool = skip_train
        self.data_path: list[str] = data_path
        self.split: str = split
        self.train_data_path: list[str] = train_data_path
        self.valid_data_path: list[str] = valid_data_path
        self.test_data_path: list[str] = test_data_path
        self.data_args_path: str = data_args_path
        self.per_split_data_args_path: str = per_split_data_args_path
        self.data_cache_path: None = data_cache_path
        self.mmap_bin_files: bool = mmap_bin_files
        self.mock_data: bool = mock_data
        self.seq_length: int = seq_length
        self.encoder_seq_length: int = encoder_seq_length
        self.decoder_seq_length: int = decoder_seq_length
        self.retriever_seq_length: int = retriever_seq_length
        self.sample_rate: float = sample_rate
        self.mask_prob: float = mask_prob
        self.short_seq_prob: float = short_seq_prob
        self.num_workers: int = num_workers
        self.reset_position_ids: bool = reset_position_ids
        self.reset_attention_mask: bool = reset_attention_mask
        self.eod_mask_loss: bool = eod_mask_loss
        self.create_attention_mask_in_dataloader: bool = (
            create_attention_mask_in_dataloader
        )
        self.num_dataset_builder_threads: int = num_dataset_builder_threads
        self.object_storage_cache_path: str = object_storage_cache_path
        self.mid_level_dataset_surplus: float = mid_level_dataset_surplus
        self.vocab_size: int = vocab_size
        self.vocab_file: str = vocab_file
        self.merge_file: str = merge_file
        self.vocab_extra_ids: int = vocab_extra_ids
        self.tokenizer_type: str = tokenizer_type
        self.tokenizer_model: str = tokenizer_model
        self.tiktoken_pattern: str = tiktoken_pattern
        self.tiktoken_num_special_tokens: int = tiktoken_num_special_tokens
        self.tiktoken_special_tokens: list[str] = tiktoken_special_tokens
        self.adlr_autoresume: bool = adlr_autoresume
        self.adlr_autoresume_interval: int = adlr_autoresume_interval
        self.ict_head_size: int = ict_head_size
        self.biencoder_projection_dim: int = biencoder_projection_dim
        self.biencoder_shared_query_context_model: bool = (
            biencoder_shared_query_context_model
        )
        self.ict_load: str = ict_load
        self.bert_load: str = bert_load
        self.titles_data_path: str = titles_data_path
        self.query_in_block_prob: float = query_in_block_prob
        self.use_one_sent_docs: bool = use_one_sent_docs
        self.evidence_data_path: str = evidence_data_path
        self.retriever_report_topk_accuracies: list[int] = (
            retriever_report_topk_accuracies
        )
        self.retriever_score_scaling: bool = retriever_score_scaling
        self.block_data_path: str = block_data_path
        self.embedding_path: str = embedding_path
        self.indexer_batch_size: int = indexer_batch_size
        self.indexer_log_interval: int = indexer_log_interval
        self.num_classes: int = num_classes
        self.img_h: int = img_h
        self.img_w: int = img_w
        self.num_channels: int = num_channels
        self.patch_dim: int = patch_dim
        self.classes_fraction: float = classes_fraction
        self.data_per_class_fraction: float = data_per_class_fraction
        self.data_sharding: bool = data_sharding
        self.head_lr_mult: float = head_lr_mult
        self.vision_pretraining: bool = vision_pretraining
        self.vision_pretraining_type: str = vision_pretraining_type
        self.vision_backbone_type: str = vision_backbone_type
        self.swin_backbone_type: str = swin_backbone_type
        self.mask_type: str = mask_type
        self.mask_factor: float = mask_factor
        self.iter_per_epoch: int = iter_per_epoch
        self.dino_local_img_size: int = dino_local_img_size
        self.dino_local_crops_number: int = dino_local_crops_number
        self.dino_head_hidden_size: int = dino_head_hidden_size
        self.dino_bottleneck_size: int = dino_bottleneck_size
        self.dino_freeze_last_layer: float = dino_freeze_last_layer
        self.dino_norm_last_layer: bool = dino_norm_last_layer
        self.dino_warmup_teacher_temp: float = dino_warmup_teacher_temp
        self.dino_teacher_temp: float = dino_teacher_temp
        self.dino_warmup_teacher_temp_epochs: int = dino_warmup_teacher_temp_epochs
        self.qk_layernorm: bool = qk_layernorm
        self.qk_l2_norm: bool = qk_l2_norm
        self.expert_model_parallel_size: int = expert_model_parallel_size
        self.expert_tensor_parallel_size: int = expert_tensor_parallel_size
        self.num_experts: int = num_experts
        self.moe_layer_freq: Any = moe_layer_freq
        self.moe_ffn_hidden_size: int = moe_ffn_hidden_size
        self.moe_shared_expert_intermediate_size: int = (
            moe_shared_expert_intermediate_size
        )
        self.moe_shared_expert_overlap: bool = moe_shared_expert_overlap
        self.moe_grouped_gemm: bool = moe_grouped_gemm
        self.moe_use_legacy_grouped_gemm: bool = moe_use_legacy_grouped_gemm
        self.moe_layer_recompute: bool = moe_layer_recompute
        self.moe_extended_tp: bool = moe_extended_tp
        self.moe_use_upcycling: bool = moe_use_upcycling
        self.moe_router_load_balancing_type: str = moe_router_load_balancing_type
        self.moe_router_dtype: str = moe_router_dtype
        self.moe_router_score_function: str = moe_router_score_function
        self.moe_router_topk: int = moe_router_topk
        self.moe_router_pre_softmax: bool = moe_router_pre_softmax
        self.moe_router_num_groups: int = moe_router_num_groups
        self.moe_router_group_topk: int = moe_router_group_topk
        self.moe_router_topk_scaling_factor: float = moe_router_topk_scaling_factor
        self.moe_router_enable_expert_bias: bool = moe_router_enable_expert_bias
        self.moe_router_bias_update_rate: float = moe_router_bias_update_rate
        self.moe_router_force_load_balancing: bool = moe_router_force_load_balancing
        self.moe_router_padding_for_fp8: bool = moe_router_padding_for_fp8
        self.moe_aux_loss_coeff: float = moe_aux_loss_coeff
        self.moe_z_loss_coeff: float = moe_z_loss_coeff
        self.moe_input_jitter_eps: float = moe_input_jitter_eps
        self.moe_per_layer_logging: bool = moe_per_layer_logging
        self.moe_token_dispatcher_type: str = moe_token_dispatcher_type
        self.moe_enable_deepep: bool = moe_enable_deepep
        self.moe_deepep_num_sms: int = moe_deepep_num_sms
        self.moe_permute_fusion: bool = moe_permute_fusion
        self.moe_expert_capacity_factor: float = moe_expert_capacity_factor
        self.moe_pad_expert_input_to_capacity: bool = moe_pad_expert_input_to_capacity
        self.moe_token_drop_policy: str = moe_token_drop_policy
        self.moe_apply_probs_on_input: bool = moe_apply_probs_on_input
        self.delay_wgrad_compute: bool = delay_wgrad_compute
        self.moe_upcycling_granularity: int = moe_upcycling_granularity
        self.q_lora_rank: int = q_lora_rank
        self.kv_lora_rank: int = kv_lora_rank
        self.qk_head_dim: int = qk_head_dim
        self.qk_pos_emb_head_dim: int = qk_pos_emb_head_dim
        self.v_head_dim: int = v_head_dim
        self.rotary_scaling_factor: float = rotary_scaling_factor
        self.mscale: float = mscale
        self.mscale_all_dim: float = mscale_all_dim
        self.heterogeneous_layers_config_path: str = heterogeneous_layers_config_path
        self.heterogeneous_layers_config_encoded_json: str = (
            heterogeneous_layers_config_encoded_json
        )
        self.log_params_norm: bool = log_params_norm
        self.log_num_zeros_in_grad: bool = log_num_zeros_in_grad
        self.log_throughput: bool = log_throughput
        self.log_progress: bool = log_progress
        self.timing_log_level: int = timing_log_level
        self.log_energy: bool = log_energy
        self.barrier_with_L1_time: bool = barrier_with_L1_time
        self.timing_log_option: str = timing_log_option
        self.tensorboard_log_interval: int = tensorboard_log_interval
        self.tensorboard_queue_size: int = tensorboard_queue_size
        self.log_timers_to_tensorboard: bool = log_timers_to_tensorboard
        self.log_loss_scale_to_tensorboard: bool = log_loss_scale_to_tensorboard
        self.log_validation_ppl_to_tensorboard: bool = log_validation_ppl_to_tensorboard
        self.log_memory_to_tensorboard: bool = log_memory_to_tensorboard
        self.log_world_size_to_tensorboard: bool = log_world_size_to_tensorboard
        self.wandb_project: str = wandb_project
        self.wandb_exp_name: str = wandb_exp_name
        self.wandb_save_dir: str = wandb_save_dir
        self.logging_level: int = logging_level
        self.log_straggler: bool = log_straggler
        self.disable_straggler_on_startup: bool = disable_straggler_on_startup
        self.straggler_ctrlr_port: int = straggler_ctrlr_port
        self.straggler_minmax_count: int = straggler_minmax_count
        self.run_workload_inspector_server: bool = run_workload_inspector_server
        self.inference_batch_times_seqlen_threshold: int = (
            inference_batch_times_seqlen_threshold
        )
        self.max_tokens_to_oom: int = max_tokens_to_oom
        self.output_bert_embeddings: bool = output_bert_embeddings
        self.bert_embedder_type: None = bert_embedder_type
        self.flash_decode: bool = flash_decode
        self.enable_cuda_graph: bool = enable_cuda_graph
        self.cuda_graph_warmup_steps: int = cuda_graph_warmup_steps
        self.external_cuda_graph: bool = external_cuda_graph
        self.cuda_graph_scope: str = cuda_graph_scope
        self.inference_max_batch_size: int = inference_max_batch_size
        self.inference_max_seq_length: int = inference_max_seq_length
        self.inference_dynamic_batching: bool = inference_dynamic_batching
        self.inference_dynamic_batching_buffer_size_gb: float = (
            inference_dynamic_batching_buffer_size_gb
        )
        self.inference_dynamic_batching_chunk_size: int = (
            inference_dynamic_batching_chunk_size
        )
        self.inference_dynamic_batching_buffer_guaranteed_fraction: float = (
            inference_dynamic_batching_buffer_guaranteed_fraction
        )
        self.inference_dynamic_batching_buffer_overflow_factor: float = (
            inference_dynamic_batching_buffer_overflow_factor
        )
        self.inference_dynamic_batching_max_requests_override: int = (
            inference_dynamic_batching_max_requests_override
        )
        self.inference_dynamic_batching_max_tokens_override: int = (
            inference_dynamic_batching_max_tokens_override
        )
        self.symmetric_ar_type: str = symmetric_ar_type
        self.nccl_all_reduce_for_prefill: bool = nccl_all_reduce_for_prefill
        self.mlp_chunks_for_prefill: int = mlp_chunks_for_prefill
        self.fp8: None = fp8
        self.fp8_recipe: None = fp8_recipe
        self.fp8_margin: int = fp8_margin
        self.fp8_interval: int = fp8_interval
        self.fp8_amax_history_len: int = fp8_amax_history_len
        self.fp8_amax_compute_algo: None = fp8_amax_compute_algo
        self.fp8_wgrad: bool = fp8_wgrad
        self.transformer_impl: None = transformer_impl
        self.fp8_param_gather: bool = fp8_param_gather
        self.first_last_layers_bf16: bool = first_last_layers_bf16
        self.num_layers_at_start_in_bf16: int = num_layers_at_start_in_bf16
        self.num_layers_at_end_in_bf16: int = num_layers_at_end_in_bf16
        self.te_rng_tracker: bool = te_rng_tracker
        self.inference_rng_tracker: bool = inference_rng_tracker
        self.retro_project_dir: None = retro_project_dir
        self.retro_add_retriever: bool = retro_add_retriever
        self.retro_cyclic_train_iters: int = retro_cyclic_train_iters
        self.retro_encoder_layers: int = retro_encoder_layers
        self.retro_encoder_hidden_dropout: float = retro_encoder_hidden_dropout
        self.retro_encoder_attention_dropout: float = retro_encoder_attention_dropout
        self.retro_num_neighbors: int = retro_num_neighbors
        self.retro_num_retrieved_chunks: int = retro_num_retrieved_chunks
        self.retro_attention_gate: float = retro_attention_gate
        self.retro_verify_neighbor_count: bool = retro_verify_neighbor_count
        self.enable_experimental: bool = enable_experimental
        self.spec: list[str] = spec
        self.hybrid_attention_ratio: float = hybrid_attention_ratio
        self.hybrid_mlp_ratio: float = hybrid_mlp_ratio
        self.hybrid_override_pattern: str = hybrid_override_pattern
        self.mamba_state_dim: int = mamba_state_dim
        self.mamba_head_dim: int = mamba_head_dim
        self.mamba_num_groups: int = mamba_num_groups
        self.mamba_num_heads: int = mamba_num_heads
        self.is_hybrid_model: bool = is_hybrid_model
        self.disable_mamba_mem_eff_path: bool = disable_mamba_mem_eff_path
        self.yaml_cfg: str = yaml_cfg
        self.use_precision_aware_optimizer: bool = use_precision_aware_optimizer
        self.main_grads_dtype: None = main_grads_dtype
        self.main_params_dtype: None = main_params_dtype
        self.exp_avg_dtype: None = exp_avg_dtype
        self.exp_avg_sq_dtype: None = exp_avg_sq_dtype
        self.enable_one_logger: bool = enable_one_logger
        self.one_logger_project: str = one_logger_project
        self.one_logger_run_name: str = one_logger_run_name
        self.one_logger_async: bool = one_logger_async
        self.app_tag_run_name: str = app_tag_run_name
        self.app_tag_run_version: str = app_tag_run_version
        self.inprocess_restart: bool = inprocess_restart
        self.inprocess_max_iterations: int = inprocess_max_iterations
        self.inprocess_monitor_thread_interval: float = (
            inprocess_monitor_thread_interval
        )
        self.inprocess_monitor_process_interval: float = (
            inprocess_monitor_process_interval
        )
        self.inprocess_progress_watchdog_interval: float = (
            inprocess_progress_watchdog_interval
        )
        self.inprocess_heartbeat_interval: float = inprocess_heartbeat_interval
        self.inprocess_soft_timeout: float = inprocess_soft_timeout
        self.inprocess_hard_timeout: float = inprocess_hard_timeout
        self.inprocess_heartbeat_timeout: float = inprocess_heartbeat_timeout
        self.inprocess_barrier_timeout: float = inprocess_barrier_timeout
        self.inprocess_completion_timeout: float = inprocess_completion_timeout
        self.inprocess_last_call_wait: float = inprocess_last_call_wait
        self.inprocess_termination_grace_time: float = inprocess_termination_grace_time
        self.inprocess_granularity: str = inprocess_granularity
        self.inprocess_active_world_size: int = inprocess_active_world_size
        self.inprocess_empty_cuda_cache: bool = inprocess_empty_cuda_cache
        self.enable_ft_package: bool = enable_ft_package
        self.calc_ft_timeouts: bool = calc_ft_timeouts
        self.config_logger_dir: str = config_logger_dir
        self.error_injection_rate: int = error_injection_rate
        self.error_injection_type: str = error_injection_type
        self.rerun_mode: str = rerun_mode
        self.enable_msc: bool = enable_msc
        self.kitchen_config_file: str = kitchen_config_file
        self.kitchen_recipe_number: int = kitchen_recipe_number
        self.sft: bool = sft
        self.sft_tokenizer_prompt_format: str = sft_tokenizer_prompt_format
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id
        self.eos_token_id: int = eos_token_id
        self.use_cache: bool = use_cache
        self.tie_word_embeddings: bool = tie_word_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MegatronConfig"]
