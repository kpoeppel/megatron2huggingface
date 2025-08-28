"""Megatron model configuration - generated from megatron2huggingface"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class MegatronConfig(PretrainedConfig):
    r"""
    This configures a MegatronModel.

    Args:
        num_layers (`int`):
            Argument num_layers.
        encoder_num_layers (`int`):
            Argument encoder_num_layers.
        decoder_num_layers (`int`):
            Argument decoder_num_layers.
        hidden_size (`int`):
            Argument hidden_size.
        ffn_hidden_size (`int`):
            Argument ffn_hidden_size.
        num_attention_heads (`int`):
            Argument num_attention_heads.
        attention_backend (`str`, *optional*, defaults to "default"):
            Argument attention_backend.
        kv_channels (`int`):
            Argument kv_channels.
        group_query_attention (`bool`, *optional*, defaults to `False`):
            Argument group_query_attention.
        num_query_groups (`int`, *optional*, defaults to 1):
            Argument num_query_groups.
        max_position_embeddings (`int`):
            Argument max_position_embeddings.
        position_embedding_type (`str`, *optional*, defaults to "learned_absolute"):
            Argument position_embedding_type.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Argument relative_attention_num_buckets.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            Argument relative_attention_max_distance.
        use_rotary_position_embeddings (`bool`, *optional*, defaults to `False`):
            Argument use_rotary_position_embeddings.
        rotary_base (`int`, *optional*, defaults to 10000):
            Argument rotary_base.
        rotary_percent (`float`, *optional*, defaults to 1.0):
            Argument rotary_percent.
        rotary_interleaved (`bool`, *optional*, defaults to `False`):
            Argument rotary_interleaved.
        rotary_seq_len_interpolation_factor (`int`):
            Argument rotary_seq_len_interpolation_factor.
        use_rope_scaling (`bool`, *optional*, defaults to `False`):
            Argument use_rope_scaling.
        rope_scaling_factor (`float`, *optional*, defaults to 8.0):
            Argument rope_scaling_factor.
        no_rope_freq (Any):
            Argument no_rope_freq.
        add_position_embedding (`bool`, *optional*, defaults to `True`):
            Argument add_position_embedding.
        mrope_section (`int`):
            Argument mrope_section.
        make_vocab_size_divisible_by (`int`, *optional*, defaults to 128):
            Argument make_vocab_size_divisible_by.
        normalization (None, *optional*, defaults to "LayerNorm"):
            Argument normalization.
        norm_epsilon (`float`, *optional*, defaults to 1e-05):
            Argument norm_epsilon.
        apply_layernorm_1p (`bool`, *optional*, defaults to `False`):
            Argument apply_layernorm_1p.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            Argument apply_residual_connection_post_layernorm.
        openai_gelu (`bool`, *optional*, defaults to `False`):
            Argument openai_gelu.
        squared_relu (`bool`, *optional*, defaults to `False`):
            Argument squared_relu.
        swiglu (`bool`, *optional*, defaults to `False`):
            Argument swiglu.
        onnx_safe (`bool`):
            Argument onnx_safe.
        bert_binary_head (`bool`, *optional*, defaults to `True`):
            Argument bert_binary_head.
        untie_embeddings_and_output_weights (`bool`, *optional*, defaults to `False`):
            Argument untie_embeddings_and_output_weights.
        multi_latent_attention (`bool`, *optional*, defaults to `False`):
            Argument multi_latent_attention.
        mtp_num_layers (`int`):
            Argument mtp_num_layers.
        mtp_loss_scaling_factor (`float`, *optional*, defaults to 0.1):
            Argument mtp_loss_scaling_factor.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Argument attention_dropout.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Argument hidden_dropout.
        weight_decay (`float`, *optional*, defaults to 0.01):
            Argument weight_decay.
        start_weight_decay (`float`):
            Argument start_weight_decay.
        end_weight_decay (`float`):
            Argument end_weight_decay.
        weight_decay_incr_style (`str`, *optional*, defaults to "constant"):
            Argument weight_decay_incr_style.
        clip_grad (`float`, *optional*, defaults to 1.0):
            Argument clip_grad.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            Argument adam_beta1.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            Argument adam_beta2.
        adam_eps (`float`, *optional*, defaults to 1e-08):
            Argument adam_eps.
        sgd_momentum (`float`, *optional*, defaults to 0.9):
            Argument sgd_momentum.
        micro_batch_size (`int`):
            Argument micro_batch_size.
        batch_size (`int`):
            Argument batch_size.
        global_batch_size (`int`):
            Argument global_batch_size.
        rampup_batch_size (None):
            Argument rampup_batch_size.
        decrease_batch_size_if_needed (`bool`, *optional*, defaults to `False`):
            Argument decrease_batch_size_if_needed.
        recompute_activations (`bool`, *optional*, defaults to `False`):
            Argument recompute_activations.
        recompute_granularity (`str`):
            Argument recompute_granularity.
        check_for_nan_in_loss_and_grad (`bool`, *optional*, defaults to `True`):
            Argument check_for_nan_in_loss_and_grad.
        check_for_spiky_loss (`bool`, *optional*, defaults to `False`):
            Argument check_for_spiky_loss.
        check_for_large_grads (`bool`, *optional*, defaults to `False`):
            Argument check_for_large_grads.
        distribute_saved_activations (`bool`, *optional*, defaults to `False`):
            Argument distribute_saved_activations.
        recompute_method (`str`):
            Argument recompute_method.
        recompute_num_layers (`int`):
            Argument recompute_num_layers.
        recompute_modules (`str`):
            Argument recompute_modules.
        clone_scatter_output_in_embedding (`bool`, *optional*, defaults to `True`):
            Argument clone_scatter_output_in_embedding.
        profile (`bool`, *optional*, defaults to `False`):
            Argument profile.
        profile_step_start (`int`, *optional*, defaults to 10):
            Argument profile_step_start.
        profile_step_end (`int`, *optional*, defaults to 12):
            Argument profile_step_end.
        iterations_to_skip (`int`, *optional*, defaults to []):
            Argument iterations_to_skip.
        result_rejected_tracker_filename (`str`):
            Argument result_rejected_tracker_filename.
        enable_gloo_process_groups (`bool`, *optional*, defaults to `True`):
            Argument enable_gloo_process_groups.
        use_pytorch_profiler (`bool`, *optional*, defaults to `False`):
            Argument use_pytorch_profiler.
        profile_ranks (`int`, *optional*, defaults to [0]):
            Argument profile_ranks.
        record_memory_history (`bool`, *optional*, defaults to `False`):
            Argument record_memory_history.
        memory_snapshot_path (`str`, *optional*, defaults to "snapshot.pickle"):
            Argument memory_snapshot_path.
        tp_comm_overlap (`bool`, *optional*, defaults to `False`):
            Argument tp_comm_overlap.
        tp_comm_overlap_cfg (`str`):
            Argument tp_comm_overlap_cfg.
        tp_comm_overlap_ag (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_overlap_ag.
        tp_comm_overlap_rs (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_overlap_rs.
        tp_comm_overlap_rs_dgrad (`bool`, *optional*, defaults to `False`):
            Argument tp_comm_overlap_rs_dgrad.
        tp_comm_bulk_dgrad (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_bulk_dgrad.
        tp_comm_bulk_wgrad (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_bulk_wgrad.
        tp_comm_bootstrap_backend (`str`, *optional*, defaults to "nccl"):
            Argument tp_comm_bootstrap_backend.
        use_cpu_initialization (`bool`):
            Argument use_cpu_initialization.
        empty_unused_memory_level (`int`, *optional*, defaults to 0):
            Argument empty_unused_memory_level.
        deterministic_mode (`bool`, *optional*, defaults to `False`):
            Argument deterministic_mode.
        check_weight_hash_across_dp_replicas_interval (`int`):
            Argument check_weight_hash_across_dp_replicas_interval.
        calculate_per_token_loss (`bool`, *optional*, defaults to `False`):
            Argument calculate_per_token_loss.
        train_sync_interval (`int`):
            Argument train_sync_interval.
        checkpoint_activations (`bool`, *optional*, defaults to `False`):
            Argument checkpoint_activations.
        train_iters (`int`):
            Argument train_iters.
        train_samples (`int`):
            Argument train_samples.
        log_interval (`int`, *optional*, defaults to 100):
            Argument log_interval.
        exit_interval (`int`):
            Argument exit_interval.
        exit_duration_in_mins (`int`):
            Argument exit_duration_in_mins.
        exit_signal_handler (`bool`, *optional*, defaults to `False`):
            Argument exit_signal_handler.
        tensorboard_dir (`str`):
            Argument tensorboard_dir.
        masked_softmax_fusion (`bool`, *optional*, defaults to `True`):
            Argument masked_softmax_fusion.
        bias_gelu_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_gelu_fusion.
        bias_swiglu_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_swiglu_fusion.
        bias_dropout_fusion (`bool`, *optional*, defaults to `True`):
            Argument bias_dropout_fusion.
        apply_rope_fusion (`bool`, *optional*, defaults to `True`):
            Argument apply_rope_fusion.
        cross_entropy_loss_fusion (`bool`, *optional*, defaults to `False`):
            Argument cross_entropy_loss_fusion.
        cross_entropy_fusion_impl (`str`, *optional*, defaults to "native"):
            Argument cross_entropy_fusion_impl.
        use_flash_attn (`bool`, *optional*, defaults to `False`):
            Argument use_flash_attn.
        add_bias_linear (`bool`, *optional*, defaults to `True`):
            Argument add_bias_linear.
        add_qkv_bias (`bool`, *optional*, defaults to `False`):
            Argument add_qkv_bias.
        optimizer (`str`, *optional*, defaults to "adam"):
            Argument optimizer.
        optimizer_cpu_offload (`bool`, *optional*, defaults to `False`):
            Argument optimizer_cpu_offload.
        optimizer_offload_fraction (`float`, *optional*, defaults to 1.0):
            Argument optimizer_offload_fraction.
        use_torch_optimizer_for_cpu_offload (`bool`, *optional*, defaults to `False`):
            Argument use_torch_optimizer_for_cpu_offload.
        overlap_cpu_optimizer_d2h_h2d (`bool`, *optional*, defaults to `False`):
            Argument overlap_cpu_optimizer_d2h_h2d.
        pin_cpu_grads (`bool`, *optional*, defaults to `True`):
            Argument pin_cpu_grads.
        pin_cpu_params (`bool`, *optional*, defaults to `True`):
            Argument pin_cpu_params.
        dataloader_type (`str`):
            Argument dataloader_type.
        async_tensor_model_parallel_allreduce (`bool`, *optional*, defaults to `True`):
            Argument async_tensor_model_parallel_allreduce.
        no_persist_layer_norm (`bool`, *optional*, defaults to `False`):
            Argument no_persist_layer_norm.
        sequence_parallel (`bool`, *optional*, defaults to `False`):
            Argument sequence_parallel.
        gradient_accumulation_fusion (`bool`, *optional*, defaults to `True`):
            Argument gradient_accumulation_fusion.
        deprecated_use_mcore_models (`bool`, *optional*, defaults to `False`):
            Argument deprecated_use_mcore_models.
        use_legacy_models (`bool`, *optional*, defaults to `False`):
            Argument use_legacy_models.
        manual_gc (`bool`, *optional*, defaults to `False`):
            Argument manual_gc.
        manual_gc_interval (`int`, *optional*, defaults to 0):
            Argument manual_gc_interval.
        manual_gc_eval (`bool`, *optional*, defaults to `True`):
            Argument manual_gc_eval.
        tp_comm_split_ag (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_split_ag.
        tp_comm_split_rs (`bool`, *optional*, defaults to `True`):
            Argument tp_comm_split_rs.
        pipeline_model_parallel_comm_backend (`str`):
            Argument pipeline_model_parallel_comm_backend.
        high_priority_stream_groups (`str`, *optional*, defaults to []):
            Argument high_priority_stream_groups.
        seed (`int`, *optional*, defaults to 1234):
            Argument seed.
        data_parallel_random_init (`bool`, *optional*, defaults to `False`):
            Argument data_parallel_random_init.
        init_method_std (`float`, *optional*, defaults to 0.02):
            Argument init_method_std.
        init_method_xavier_uniform (`bool`, *optional*, defaults to `False`):
            Argument init_method_xavier_uniform.
        lr (`float`):
            Argument lr.
        lr_decay_style (`str`, *optional*, defaults to "linear"):
            Argument lr_decay_style.
        lr_wsd_decay_style (`str`, *optional*, defaults to "exponential"):
            Argument lr_wsd_decay_style.
        lr_decay_iters (`int`):
            Argument lr_decay_iters.
        lr_decay_samples (`int`):
            Argument lr_decay_samples.
        lr_wsd_decay_samples (`int`):
            Argument lr_wsd_decay_samples.
        lr_wsd_decay_iters (`int`):
            Argument lr_wsd_decay_iters.
        lr_warmup_fraction (`float`):
            Argument lr_warmup_fraction.
        lr_warmup_iters (`int`, *optional*, defaults to 0):
            Argument lr_warmup_iters.
        lr_warmup_samples (`int`, *optional*, defaults to 0):
            Argument lr_warmup_samples.
        lr_warmup_init (`float`, *optional*, defaults to 0.0):
            Argument lr_warmup_init.
        warmup (`int`):
            Argument warmup.
        min_lr (`float`, *optional*, defaults to 0.0):
            Argument min_lr.
        override_opt_param_scheduler (`bool`, *optional*, defaults to `False`):
            Argument override_opt_param_scheduler.
        use_checkpoint_opt_param_scheduler (`bool`, *optional*, defaults to `False`):
            Argument use_checkpoint_opt_param_scheduler.
        decoupled_lr (`float`):
            Argument decoupled_lr.
        decoupled_min_lr (`float`):
            Argument decoupled_min_lr.
        save (`str`):
            Argument save.
        save_interval (`int`):
            Argument save_interval.
        no_save_optim (`bool`):
            Argument no_save_optim.
        no_save_rng (`bool`):
            Argument no_save_rng.
        load (`str`):
            Argument load.
        no_load_optim (`bool`):
            Argument no_load_optim.
        no_load_rng (`bool`):
            Argument no_load_rng.
        non_persistent_save_interval (`int`):
            Argument non_persistent_save_interval.
        non_persistent_ckpt_type (`str`):
            Argument non_persistent_ckpt_type.
        non_persistent_global_ckpt_dir (`str`):
            Argument non_persistent_global_ckpt_dir.
        non_persistent_local_ckpt_dir (`str`):
            Argument non_persistent_local_ckpt_dir.
        non_persistent_local_ckpt_algo (`str`, *optional*, defaults to "fully_parallel"):
            Argument non_persistent_local_ckpt_algo.
        finetune (`bool`, *optional*, defaults to `False`):
            Argument finetune.
        pretrained_checkpoint (`str`):
            Argument pretrained_checkpoint.
        ckpt_step (`int`):
            Argument ckpt_step.
        perform_initialization (`bool`, *optional*, defaults to `True`):
            Argument perform_initialization.
        use_checkpoint_args (`bool`, *optional*, defaults to `False`):
            Argument use_checkpoint_args.
        use_mp_args_from_checkpoint_args (`bool`, *optional*, defaults to `False`):
            Argument use_mp_args_from_checkpoint_args.
        use_tokenizer_model_from_checkpoint_args (`bool`, *optional*, defaults to `True`):
            Argument use_tokenizer_model_from_checkpoint_args.
        exit_on_missing_checkpoint (`bool`, *optional*, defaults to `False`):
            Argument exit_on_missing_checkpoint.
        use_dist_ckpt_deprecated (`bool`, *optional*, defaults to `False`):
            Argument use_dist_ckpt_deprecated.
        use_persistent_ckpt_worker (`bool`, *optional*, defaults to `False`):
            Argument use_persistent_ckpt_worker.
        auto_detect_ckpt_format (`bool`, *optional*, defaults to `False`):
            Argument auto_detect_ckpt_format.
        dist_ckpt_format_deprecated (None):
            Argument dist_ckpt_format_deprecated.
        ckpt_format (None, *optional*, defaults to "torch_dist"):
            Argument ckpt_format.
        ckpt_convert_format (None):
            Argument ckpt_convert_format.
        ckpt_convert_save (None):
            Argument ckpt_convert_save.
        ckpt_convert_update_legacy_dist_opt_format (`bool`, *optional*, defaults to `False`):
            Argument ckpt_convert_update_legacy_dist_opt_format.
        ckpt_fully_parallel_save_deprecated (`bool`, *optional*, defaults to `False`):
            Argument ckpt_fully_parallel_save_deprecated.
        ckpt_fully_parallel_save (`bool`, *optional*, defaults to `True`):
            Argument ckpt_fully_parallel_save.
        async_save (`bool`):
            Argument async_save.
        ckpt_fully_parallel_load (`bool`, *optional*, defaults to `False`):
            Argument ckpt_fully_parallel_load.
        ckpt_assume_constant_structure (`bool`, *optional*, defaults to `False`):
            Argument ckpt_assume_constant_structure.
        dist_ckpt_strictness (`str`, *optional*, defaults to "assume_ok_unexpected"):
            Argument dist_ckpt_strictness.
        load_model_opt_format (`bool`, *optional*, defaults to `False`):
            Argument load_model_opt_format.
        fp16 (`bool`, *optional*, defaults to `False`):
            Argument fp16.
        bf16 (`bool`, *optional*, defaults to `False`):
            Argument bf16.
        grad_reduce_in_bf16 (`bool`, *optional*, defaults to `False`):
            Argument grad_reduce_in_bf16.
        loss_scale (`float`):
            Argument loss_scale.
        initial_loss_scale (`float`, *optional*, defaults to 4294967296):
            Argument initial_loss_scale.
        min_loss_scale (`float`, *optional*, defaults to 1.0):
            Argument min_loss_scale.
        loss_scale_window (`float`, *optional*, defaults to 1000):
            Argument loss_scale_window.
        hysteresis (`int`, *optional*, defaults to 2):
            Argument hysteresis.
        fp32_residual_connection (`bool`, *optional*, defaults to `False`):
            Argument fp32_residual_connection.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `False`):
            Argument apply_query_key_layer_scaling.
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `False`):
            Argument attention_softmax_in_fp32.
        accumulate_allreduce_grads_in_fp32 (`bool`, *optional*, defaults to `False`):
            Argument accumulate_allreduce_grads_in_fp32.
        fp16_lm_cross_entropy (`bool`, *optional*, defaults to `False`):
            Argument fp16_lm_cross_entropy.
        disable_bf16_reduced_precision_matmul (`bool`, *optional*, defaults to `False`):
            Argument disable_bf16_reduced_precision_matmul.
        reuse_grad_buf_for_mxfp8_param_ag (`bool`, *optional*, defaults to `False`):
            Argument reuse_grad_buf_for_mxfp8_param_ag.
        tensor_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument tensor_model_parallel_size.
        encoder_tensor_model_parallel_size (`int`, *optional*, defaults to 0):
            Argument encoder_tensor_model_parallel_size.
        pipeline_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument pipeline_model_parallel_size.
        encoder_pipeline_model_parallel_size (`int`, *optional*, defaults to 0):
            Argument encoder_pipeline_model_parallel_size.
        pipeline_model_parallel_split_rank (`int`):
            Argument pipeline_model_parallel_split_rank.
        decoder_first_pipeline_num_layers (`int`):
            Argument decoder_first_pipeline_num_layers.
        decoder_last_pipeline_num_layers (`int`):
            Argument decoder_last_pipeline_num_layers.
        pipeline_model_parallel_layout (`str`):
            Argument pipeline_model_parallel_layout.
        model_parallel_size (`int`):
            Argument model_parallel_size.
        num_layers_per_virtual_pipeline_stage (`int`):
            Argument num_layers_per_virtual_pipeline_stage.
        num_virtual_stages_per_pipeline_rank (`int`):
            Argument num_virtual_stages_per_pipeline_rank.
        microbatch_group_size_per_vp_stage (`int`):
            Argument microbatch_group_size_per_vp_stage.
        overlap_p2p_comm (`bool`, *optional*, defaults to `True`):
            Argument overlap_p2p_comm.
        overlap_p2p_comm_warmup_flush (`bool`, *optional*, defaults to `False`):
            Argument overlap_p2p_comm_warmup_flush.
        distributed_backend (None, *optional*, defaults to "nccl"):
            Argument distributed_backend.
        distributed_timeout_minutes (`int`, *optional*, defaults to 10):
            Argument distributed_timeout_minutes.
        overlap_grad_reduce (`bool`, *optional*, defaults to `False`):
            Argument overlap_grad_reduce.
        defer_embedding_wgrad_compute (`bool`, *optional*, defaults to `False`):
            Argument defer_embedding_wgrad_compute.
        wgrad_deferral_limit (`int`, *optional*, defaults to 0):
            Argument wgrad_deferral_limit.
        align_grad_reduce (`bool`, *optional*, defaults to `True`):
            Argument align_grad_reduce.
        ddp_num_buckets (`int`):
            Argument ddp_num_buckets.
        ddp_bucket_size (`int`):
            Argument ddp_bucket_size.
        ddp_pad_buckets_for_high_nccl_busbw (`bool`, *optional*, defaults to `False`):
            Argument ddp_pad_buckets_for_high_nccl_busbw.
        ddp_average_in_collective (`bool`, *optional*, defaults to `False`):
            Argument ddp_average_in_collective.
        overlap_param_gather (`bool`, *optional*, defaults to `False`):
            Argument overlap_param_gather.
        overlap_param_gather_with_optimizer_step (`bool`, *optional*, defaults to `False`):
            Argument overlap_param_gather_with_optimizer_step.
        align_param_gather (`bool`, *optional*, defaults to `True`):
            Argument align_param_gather.
        scatter_gather_tensors_in_pipeline (`bool`, *optional*, defaults to `True`):
            Argument scatter_gather_tensors_in_pipeline.
        use_ring_exchange_p2p (`bool`, *optional*, defaults to `False`):
            Argument use_ring_exchange_p2p.
        local_rank (`int`, *optional*, defaults to 0):
            Argument local_rank.
        lazy_mpu_init (`bool`):
            Argument lazy_mpu_init.
        account_for_embedding_in_pipeline_split (`bool`, *optional*, defaults to `False`):
            Argument account_for_embedding_in_pipeline_split.
        account_for_loss_in_pipeline_split (`bool`, *optional*, defaults to `False`):
            Argument account_for_loss_in_pipeline_split.
        use_distributed_optimizer (`bool`, *optional*, defaults to `False`):
            Argument use_distributed_optimizer.
        nccl_ub (`bool`, *optional*, defaults to `False`):
            Argument nccl_ub.
        use_sharp (`bool`, *optional*, defaults to `False`):
            Argument use_sharp.
        use_custom_fsdp (`bool`, *optional*, defaults to `False`):
            Argument use_custom_fsdp.
        init_model_with_meta_device (`bool`, *optional*, defaults to `False`):
            Argument init_model_with_meta_device.
        data_parallel_sharding_strategy (`str`, *optional*, defaults to "no_shard"):
            Argument data_parallel_sharding_strategy.
        gradient_reduce_div_fusion (`bool`, *optional*, defaults to `True`):
            Argument gradient_reduce_div_fusion.
        fsdp_double_buffer (`bool`, *optional*, defaults to `False`):
            Argument fsdp_double_buffer.
        suggested_communication_unit_size (`int`):
            Argument suggested_communication_unit_size.
        keep_fp8_transpose_cache_when_using_custom_fsdp (`bool`, *optional*, defaults to `False`):
            Argument keep_fp8_transpose_cache_when_using_custom_fsdp.
        num_distributed_optimizer_instances (`int`, *optional*, defaults to 1):
            Argument num_distributed_optimizer_instances.
        use_torch_fsdp2 (`bool`, *optional*, defaults to `False`):
            Argument use_torch_fsdp2.
        torch_fsdp2_reshard_after_forward (`bool`, *optional*, defaults to `True`):
            Argument torch_fsdp2_reshard_after_forward.
        context_parallel_size (`int`, *optional*, defaults to 1):
            Argument context_parallel_size.
        cp_comm_type (`str`, *optional*, defaults to ['p2p']):
            Argument cp_comm_type.
        hierarchical_context_parallel_sizes (`int`):
            Argument hierarchical_context_parallel_sizes.
        nccl_communicator_config_path (`str`):
            Argument nccl_communicator_config_path.
        use_tp_pp_dp_mapping (`bool`, *optional*, defaults to `False`):
            Argument use_tp_pp_dp_mapping.
        replication (`bool`, *optional*, defaults to `False`):
            Argument replication.
        replication_jump (`int`):
            Argument replication_jump.
        replication_factor (`int`, *optional*, defaults to 2):
            Argument replication_factor.
        eval_iters (`int`, *optional*, defaults to 100):
            Argument eval_iters.
        eval_interval (`int`, *optional*, defaults to 1000):
            Argument eval_interval.
        test_mode (`bool`, *optional*, defaults to `False`):
            Argument test_mode.
        skip_train (`bool`, *optional*, defaults to `False`):
            Argument skip_train.
        data_path (None):
            Argument data_path.
        split (`str`):
            Argument split.
        train_data_path (None):
            Argument train_data_path.
        valid_data_path (None):
            Argument valid_data_path.
        test_data_path (None):
            Argument test_data_path.
        data_args_path (`str`):
            Argument data_args_path.
        per_split_data_args_path (`str`):
            Argument per_split_data_args_path.
        data_cache_path (None):
            Argument data_cache_path.
        mmap_bin_files (`bool`, *optional*, defaults to `True`):
            Argument mmap_bin_files.
        mock_data (`bool`, *optional*, defaults to `False`):
            Argument mock_data.
        seq_length (`int`):
            Argument seq_length.
        encoder_seq_length (`int`):
            Argument encoder_seq_length.
        decoder_seq_length (`int`):
            Argument decoder_seq_length.
        retriever_seq_length (`int`, *optional*, defaults to 256):
            Argument retriever_seq_length.
        sample_rate (`float`, *optional*, defaults to 1.0):
            Argument sample_rate.
        mask_prob (`float`, *optional*, defaults to 0.15):
            Argument mask_prob.
        short_seq_prob (`float`, *optional*, defaults to 0.1):
            Argument short_seq_prob.
        num_workers (`int`, *optional*, defaults to 2):
            Argument num_workers.
        reset_position_ids (`bool`, *optional*, defaults to `False`):
            Argument reset_position_ids.
        reset_attention_mask (`bool`, *optional*, defaults to `False`):
            Argument reset_attention_mask.
        eod_mask_loss (`bool`, *optional*, defaults to `False`):
            Argument eod_mask_loss.
        create_attention_mask_in_dataloader (`bool`, *optional*, defaults to `True`):
            Argument create_attention_mask_in_dataloader.
        num_dataset_builder_threads (`int`, *optional*, defaults to 1):
            Argument num_dataset_builder_threads.
        object_storage_cache_path (`str`):
            Argument object_storage_cache_path.
        mid_level_dataset_surplus (`float`, *optional*, defaults to 0.005):
            Argument mid_level_dataset_surplus.
        vocab_size (`int`):
            Argument vocab_size.
        vocab_file (`str`):
            Argument vocab_file.
        merge_file (`str`):
            Argument merge_file.
        vocab_extra_ids (`int`, *optional*, defaults to 0):
            Argument vocab_extra_ids.
        tokenizer_type (`str`):
            Argument tokenizer_type.
        tokenizer_model (`str`):
            Argument tokenizer_model.
        tiktoken_pattern (`str`):
            Argument tiktoken_pattern.
        tiktoken_num_special_tokens (`int`, *optional*, defaults to 1000):
            Argument tiktoken_num_special_tokens.
        tiktoken_special_tokens (`str`):
            Argument tiktoken_special_tokens.
        adlr_autoresume (`bool`, *optional*, defaults to `False`):
            Argument adlr_autoresume.
        adlr_autoresume_interval (`int`, *optional*, defaults to 1000):
            Argument adlr_autoresume_interval.
        ict_head_size (`int`):
            Argument ict_head_size.
        biencoder_projection_dim (`int`, *optional*, defaults to 0):
            Argument biencoder_projection_dim.
        biencoder_shared_query_context_model (`bool`, *optional*, defaults to `False`):
            Argument biencoder_shared_query_context_model.
        ict_load (`str`):
            Argument ict_load.
        bert_load (`str`):
            Argument bert_load.
        titles_data_path (`str`):
            Argument titles_data_path.
        query_in_block_prob (`float`, *optional*, defaults to 0.1):
            Argument query_in_block_prob.
        use_one_sent_docs (`bool`, *optional*, defaults to `False`):
            Argument use_one_sent_docs.
        evidence_data_path (`str`):
            Argument evidence_data_path.
        retriever_report_topk_accuracies (`int`, *optional*, defaults to []):
            Argument retriever_report_topk_accuracies.
        retriever_score_scaling (`bool`, *optional*, defaults to `False`):
            Argument retriever_score_scaling.
        block_data_path (`str`):
            Argument block_data_path.
        embedding_path (`str`):
            Argument embedding_path.
        indexer_batch_size (`int`, *optional*, defaults to 128):
            Argument indexer_batch_size.
        indexer_log_interval (`int`, *optional*, defaults to 1000):
            Argument indexer_log_interval.
        num_classes (`int`, *optional*, defaults to 1000):
            Argument num_classes.
        img_h (`int`, *optional*, defaults to 224):
            Argument img_h.
        img_w (`int`, *optional*, defaults to 224):
            Argument img_w.
        num_channels (`int`, *optional*, defaults to 3):
            Argument num_channels.
        patch_dim (`int`, *optional*, defaults to 16):
            Argument patch_dim.
        classes_fraction (`float`, *optional*, defaults to 1.0):
            Argument classes_fraction.
        data_per_class_fraction (`float`, *optional*, defaults to 1.0):
            Argument data_per_class_fraction.
        data_sharding (`bool`, *optional*, defaults to `True`):
            Argument data_sharding.
        head_lr_mult (`float`, *optional*, defaults to 1.0):
            Argument head_lr_mult.
        vision_pretraining (`bool`, *optional*, defaults to `False`):
            Argument vision_pretraining.
        vision_pretraining_type (`str`, *optional*, defaults to "classify"):
            Argument vision_pretraining_type.
        vision_backbone_type (`str`, *optional*, defaults to "vit"):
            Argument vision_backbone_type.
        swin_backbone_type (`str`, *optional*, defaults to "tiny"):
            Argument swin_backbone_type.
        mask_type (`str`, *optional*, defaults to "random"):
            Argument mask_type.
        mask_factor (`float`, *optional*, defaults to 1.0):
            Argument mask_factor.
        iter_per_epoch (`int`, *optional*, defaults to 1250):
            Argument iter_per_epoch.
        dino_local_img_size (`int`, *optional*, defaults to 96):
            Argument dino_local_img_size.
        dino_local_crops_number (`int`, *optional*, defaults to 10):
            Argument dino_local_crops_number.
        dino_head_hidden_size (`int`, *optional*, defaults to 2048):
            Argument dino_head_hidden_size.
        dino_bottleneck_size (`int`, *optional*, defaults to 256):
            Argument dino_bottleneck_size.
        dino_freeze_last_layer (`float`, *optional*, defaults to 1):
            Argument dino_freeze_last_layer.
        dino_norm_last_layer (`bool`, *optional*, defaults to `False`):
            Argument dino_norm_last_layer.
        dino_warmup_teacher_temp (`float`, *optional*, defaults to 0.04):
            Argument dino_warmup_teacher_temp.
        dino_teacher_temp (`float`, *optional*, defaults to 0.07):
            Argument dino_teacher_temp.
        dino_warmup_teacher_temp_epochs (`int`, *optional*, defaults to 30):
            Argument dino_warmup_teacher_temp_epochs.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Argument qk_layernorm.
        qk_l2_norm (`bool`, *optional*, defaults to `False`):
            Argument qk_l2_norm.
        expert_model_parallel_size (`int`, *optional*, defaults to 1):
            Argument expert_model_parallel_size.
        expert_tensor_parallel_size (`int`):
            Argument expert_tensor_parallel_size.
        num_experts (`int`):
            Argument num_experts.
        moe_layer_freq (Any, *optional*, defaults to 1):
            Argument moe_layer_freq.
        moe_ffn_hidden_size (`int`):
            Argument moe_ffn_hidden_size.
        moe_shared_expert_intermediate_size (`int`):
            Argument moe_shared_expert_intermediate_size.
        moe_shared_expert_overlap (`bool`, *optional*, defaults to `False`):
            Argument moe_shared_expert_overlap.
        moe_grouped_gemm (`bool`, *optional*, defaults to `False`):
            Argument moe_grouped_gemm.
        moe_use_legacy_grouped_gemm (`bool`, *optional*, defaults to `False`):
            Argument moe_use_legacy_grouped_gemm.
        moe_layer_recompute (`bool`, *optional*, defaults to `False`):
            Argument moe_layer_recompute.
        moe_extended_tp (`bool`, *optional*, defaults to `False`):
            Argument moe_extended_tp.
        moe_use_upcycling (`bool`, *optional*, defaults to `False`):
            Argument moe_use_upcycling.
        moe_router_load_balancing_type (`str`, *optional*, defaults to "aux_loss"):
            Argument moe_router_load_balancing_type.
        moe_router_dtype (`str`):
            Argument moe_router_dtype.
        moe_router_score_function (`str`, *optional*, defaults to "softmax"):
            Argument moe_router_score_function.
        moe_router_topk (`int`, *optional*, defaults to 2):
            Argument moe_router_topk.
        moe_router_pre_softmax (`bool`, *optional*, defaults to `False`):
            Argument moe_router_pre_softmax.
        moe_router_num_groups (`int`):
            Argument moe_router_num_groups.
        moe_router_group_topk (`int`):
            Argument moe_router_group_topk.
        moe_router_topk_scaling_factor (`float`):
            Argument moe_router_topk_scaling_factor.
        moe_router_enable_expert_bias (`bool`, *optional*, defaults to `False`):
            Argument moe_router_enable_expert_bias.
        moe_router_bias_update_rate (`float`, *optional*, defaults to 0.001):
            Argument moe_router_bias_update_rate.
        moe_router_force_load_balancing (`bool`, *optional*, defaults to `False`):
            Argument moe_router_force_load_balancing.
        moe_router_padding_for_fp8 (`bool`, *optional*, defaults to `False`):
            Argument moe_router_padding_for_fp8.
        moe_aux_loss_coeff (`float`, *optional*, defaults to 0.0):
            Argument moe_aux_loss_coeff.
        moe_z_loss_coeff (`float`):
            Argument moe_z_loss_coeff.
        moe_input_jitter_eps (`float`):
            Argument moe_input_jitter_eps.
        moe_per_layer_logging (`bool`, *optional*, defaults to `False`):
            Argument moe_per_layer_logging.
        moe_token_dispatcher_type (`str`, *optional*, defaults to "allgather"):
            Argument moe_token_dispatcher_type.
        moe_enable_deepep (`bool`, *optional*, defaults to `False`):
            Argument moe_enable_deepep.
        moe_deepep_num_sms (`int`, *optional*, defaults to 20):
            Argument moe_deepep_num_sms.
        moe_permute_fusion (`bool`, *optional*, defaults to `False`):
            Argument moe_permute_fusion.
        moe_expert_capacity_factor (`float`):
            Argument moe_expert_capacity_factor.
        moe_pad_expert_input_to_capacity (`bool`, *optional*, defaults to `False`):
            Argument moe_pad_expert_input_to_capacity.
        moe_token_drop_policy (`str`, *optional*, defaults to "probs"):
            Argument moe_token_drop_policy.
        moe_apply_probs_on_input (`bool`, *optional*, defaults to `False`):
            Argument moe_apply_probs_on_input.
        delay_wgrad_compute (`bool`, *optional*, defaults to `False`):
            Argument delay_wgrad_compute.
        moe_upcycling_granularity (`int`, *optional*, defaults to 1):
            Argument moe_upcycling_granularity.
        q_lora_rank (`int`):
            Argument q_lora_rank.
        kv_lora_rank (`int`, *optional*, defaults to 32):
            Argument kv_lora_rank.
        qk_head_dim (`int`, *optional*, defaults to 128):
            Argument qk_head_dim.
        qk_pos_emb_head_dim (`int`, *optional*, defaults to 64):
            Argument qk_pos_emb_head_dim.
        v_head_dim (`int`, *optional*, defaults to 128):
            Argument v_head_dim.
        rotary_scaling_factor (`float`, *optional*, defaults to 1.0):
            Argument rotary_scaling_factor.
        mscale (`float`, *optional*, defaults to 1.0):
            Argument mscale.
        mscale_all_dim (`float`, *optional*, defaults to 1.0):
            Argument mscale_all_dim.
        heterogeneous_layers_config_path (`str`):
            Argument heterogeneous_layers_config_path.
        heterogeneous_layers_config_encoded_json (`str`):
            Argument heterogeneous_layers_config_encoded_json.
        log_params_norm (`bool`, *optional*, defaults to `False`):
            Argument log_params_norm.
        log_num_zeros_in_grad (`bool`, *optional*, defaults to `False`):
            Argument log_num_zeros_in_grad.
        log_throughput (`bool`, *optional*, defaults to `False`):
            Argument log_throughput.
        log_progress (`bool`, *optional*, defaults to `False`):
            Argument log_progress.
        timing_log_level (`int`, *optional*, defaults to 0):
            Argument timing_log_level.
        log_energy (`bool`, *optional*, defaults to `False`):
            Argument log_energy.
        barrier_with_L1_time (`bool`, *optional*, defaults to `True`):
            Argument barrier_with_L1_time.
        timing_log_option (`str`, *optional*, defaults to "minmax"):
            Argument timing_log_option.
        tensorboard_log_interval (`int`, *optional*, defaults to 1):
            Argument tensorboard_log_interval.
        tensorboard_queue_size (`int`, *optional*, defaults to 1000):
            Argument tensorboard_queue_size.
        log_timers_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_timers_to_tensorboard.
        log_loss_scale_to_tensorboard (`bool`, *optional*, defaults to `True`):
            Argument log_loss_scale_to_tensorboard.
        log_validation_ppl_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_validation_ppl_to_tensorboard.
        log_memory_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_memory_to_tensorboard.
        log_world_size_to_tensorboard (`bool`, *optional*, defaults to `False`):
            Argument log_world_size_to_tensorboard.
        wandb_project (`str`, *optional*, defaults to ""):
            Argument wandb_project.
        wandb_exp_name (`str`, *optional*, defaults to ""):
            Argument wandb_exp_name.
        wandb_save_dir (`str`, *optional*, defaults to ""):
            Argument wandb_save_dir.
        logging_level (`int`):
            Argument logging_level.
        log_straggler (`bool`, *optional*, defaults to `False`):
            Argument log_straggler.
        disable_straggler_on_startup (`bool`, *optional*, defaults to `False`):
            Argument disable_straggler_on_startup.
        straggler_ctrlr_port (`int`, *optional*, defaults to 65535):
            Argument straggler_ctrlr_port.
        straggler_minmax_count (`int`, *optional*, defaults to 1):
            Argument straggler_minmax_count.
        run_workload_inspector_server (`bool`, *optional*, defaults to `False`):
            Argument run_workload_inspector_server.
        inference_batch_times_seqlen_threshold (`int`, *optional*, defaults to -1):
            Argument inference_batch_times_seqlen_threshold.
        max_tokens_to_oom (`int`, *optional*, defaults to 12000):
            Argument max_tokens_to_oom.
        output_bert_embeddings (`bool`, *optional*, defaults to `False`):
            Argument output_bert_embeddings.
        bert_embedder_type (None, *optional*, defaults to "megatron"):
            Argument bert_embedder_type.
        flash_decode (`bool`, *optional*, defaults to `False`):
            Argument flash_decode.
        enable_cuda_graph (`bool`, *optional*, defaults to `False`):
            Argument enable_cuda_graph.
        cuda_graph_warmup_steps (`int`, *optional*, defaults to 3):
            Argument cuda_graph_warmup_steps.
        external_cuda_graph (`bool`, *optional*, defaults to `False`):
            Argument external_cuda_graph.
        cuda_graph_scope (`str`, *optional*, defaults to "full"):
            Argument cuda_graph_scope.
        inference_max_batch_size (`int`, *optional*, defaults to 8):
            Argument inference_max_batch_size.
        inference_max_seq_length (`int`, *optional*, defaults to 2560):
            Argument inference_max_seq_length.
        inference_dynamic_batching (`bool`, *optional*, defaults to `False`):
            Argument inference_dynamic_batching.
        inference_dynamic_batching_buffer_size_gb (`float`, *optional*, defaults to 40.0):
            Argument inference_dynamic_batching_buffer_size_gb.
        inference_dynamic_batching_chunk_size (`int`, *optional*, defaults to 256):
            Argument inference_dynamic_batching_chunk_size.
        inference_dynamic_batching_buffer_guaranteed_fraction (`float`, *optional*, defaults to 0.2):
            Argument inference_dynamic_batching_buffer_guaranteed_fraction.
        inference_dynamic_batching_buffer_overflow_factor (`float`):
            Argument inference_dynamic_batching_buffer_overflow_factor.
        inference_dynamic_batching_max_requests_override (`int`):
            Argument inference_dynamic_batching_max_requests_override.
        inference_dynamic_batching_max_tokens_override (`int`):
            Argument inference_dynamic_batching_max_tokens_override.
        symmetric_ar_type (`str`):
            Argument symmetric_ar_type.
        nccl_all_reduce_for_prefill (`bool`, *optional*, defaults to `False`):
            Argument nccl_all_reduce_for_prefill.
        mlp_chunks_for_prefill (`int`, *optional*, defaults to 1):
            Argument mlp_chunks_for_prefill.
        fp8 (None):
            Argument fp8.
        fp8_recipe (None, *optional*, defaults to "delayed"):
            Argument fp8_recipe.
        fp8_margin (`int`, *optional*, defaults to 0):
            Argument fp8_margin.
        fp8_interval (`int`, *optional*, defaults to 1):
            Argument fp8_interval.
        fp8_amax_history_len (`int`, *optional*, defaults to 1):
            Argument fp8_amax_history_len.
        fp8_amax_compute_algo (None, *optional*, defaults to "most_recent"):
            Argument fp8_amax_compute_algo.
        fp8_wgrad (`bool`, *optional*, defaults to `True`):
            Argument fp8_wgrad.
        transformer_impl (None, *optional*, defaults to "transformer_engine"):
            Argument transformer_impl.
        fp8_param_gather (`bool`, *optional*, defaults to `False`):
            Argument fp8_param_gather.
        first_last_layers_bf16 (`bool`, *optional*, defaults to `False`):
            Argument first_last_layers_bf16.
        num_layers_at_start_in_bf16 (`int`, *optional*, defaults to 1):
            Argument num_layers_at_start_in_bf16.
        num_layers_at_end_in_bf16 (`int`, *optional*, defaults to 1):
            Argument num_layers_at_end_in_bf16.
        te_rng_tracker (`bool`, *optional*, defaults to `False`):
            Argument te_rng_tracker.
        inference_rng_tracker (`bool`, *optional*, defaults to `False`):
            Argument inference_rng_tracker.
        retro_project_dir (None):
            Argument retro_project_dir.
        retro_add_retriever (`bool`, *optional*, defaults to `False`):
            Argument retro_add_retriever.
        retro_cyclic_train_iters (`int`):
            Argument retro_cyclic_train_iters.
        retro_encoder_layers (`int`, *optional*, defaults to 2):
            Argument retro_encoder_layers.
        retro_encoder_hidden_dropout (`float`, *optional*, defaults to 0.1):
            Argument retro_encoder_hidden_dropout.
        retro_encoder_attention_dropout (`float`, *optional*, defaults to 0.1):
            Argument retro_encoder_attention_dropout.
        retro_num_neighbors (`int`, *optional*, defaults to 2):
            Argument retro_num_neighbors.
        retro_num_retrieved_chunks (`int`, *optional*, defaults to 2):
            Argument retro_num_retrieved_chunks.
        retro_attention_gate (`float`, *optional*, defaults to 1):
            Argument retro_attention_gate.
        retro_verify_neighbor_count (`bool`, *optional*, defaults to `True`):
            Argument retro_verify_neighbor_count.
        enable_experimental (`bool`, *optional*, defaults to `False`):
            Argument enable_experimental.
        spec (`str`):
            Argument spec.
        hybrid_attention_ratio (`float`, *optional*, defaults to 0.0):
            Argument hybrid_attention_ratio.
        hybrid_mlp_ratio (`float`, *optional*, defaults to 0.0):
            Argument hybrid_mlp_ratio.
        hybrid_override_pattern (`str`):
            Argument hybrid_override_pattern.
        mamba_state_dim (`int`, *optional*, defaults to 128):
            Argument mamba_state_dim.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Argument mamba_head_dim.
        mamba_num_groups (`int`, *optional*, defaults to 8):
            Argument mamba_num_groups.
        mamba_num_heads (`int`):
            Argument mamba_num_heads.
        is_hybrid_model (`bool`, *optional*, defaults to `False`):
            Argument is_hybrid_model.
        disable_mamba_mem_eff_path (`bool`, *optional*, defaults to `False`):
            Argument disable_mamba_mem_eff_path.
        yaml_cfg (`str`):
            Argument yaml_cfg.
        use_precision_aware_optimizer (`bool`, *optional*, defaults to `False`):
            Argument use_precision_aware_optimizer.
        main_grads_dtype (None, *optional*, defaults to "fp32"):
            Argument main_grads_dtype.
        main_params_dtype (None, *optional*, defaults to "fp32"):
            Argument main_params_dtype.
        exp_avg_dtype (None, *optional*, defaults to "fp32"):
            Argument exp_avg_dtype.
        exp_avg_sq_dtype (None, *optional*, defaults to "fp32"):
            Argument exp_avg_sq_dtype.
        enable_one_logger (`bool`, *optional*, defaults to `True`):
            Argument enable_one_logger.
        one_logger_project (`str`, *optional*, defaults to "megatron-lm"):
            Argument one_logger_project.
        one_logger_run_name (`str`):
            Argument one_logger_run_name.
        one_logger_async (`bool`, *optional*, defaults to `False`):
            Argument one_logger_async.
        app_tag_run_name (`str`):
            Argument app_tag_run_name.
        app_tag_run_version (`str`, *optional*, defaults to "0.0.0"):
            Argument app_tag_run_version.
        inprocess_restart (`bool`, *optional*, defaults to `False`):
            Argument inprocess_restart.
        inprocess_max_iterations (`int`):
            Argument inprocess_max_iterations.
        inprocess_monitor_thread_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_monitor_thread_interval.
        inprocess_monitor_process_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_monitor_process_interval.
        inprocess_progress_watchdog_interval (`float`, *optional*, defaults to 1.0):
            Argument inprocess_progress_watchdog_interval.
        inprocess_heartbeat_interval (`float`, *optional*, defaults to 30):
            Argument inprocess_heartbeat_interval.
        inprocess_soft_timeout (`float`, *optional*, defaults to 60):
            Argument inprocess_soft_timeout.
        inprocess_hard_timeout (`float`, *optional*, defaults to 90):
            Argument inprocess_hard_timeout.
        inprocess_heartbeat_timeout (`float`, *optional*, defaults to 60):
            Argument inprocess_heartbeat_timeout.
        inprocess_barrier_timeout (`float`, *optional*, defaults to 120):
            Argument inprocess_barrier_timeout.
        inprocess_completion_timeout (`float`, *optional*, defaults to 120):
            Argument inprocess_completion_timeout.
        inprocess_last_call_wait (`float`, *optional*, defaults to 1):
            Argument inprocess_last_call_wait.
        inprocess_termination_grace_time (`float`, *optional*, defaults to 1):
            Argument inprocess_termination_grace_time.
        inprocess_granularity (`str`, *optional*, defaults to "node"):
            Argument inprocess_granularity.
        inprocess_active_world_size (`int`, *optional*, defaults to 1):
            Argument inprocess_active_world_size.
        inprocess_empty_cuda_cache (`bool`, *optional*, defaults to `False`):
            Argument inprocess_empty_cuda_cache.
        enable_ft_package (`bool`, *optional*, defaults to `False`):
            Argument enable_ft_package.
        calc_ft_timeouts (`bool`, *optional*, defaults to `False`):
            Argument calc_ft_timeouts.
        config_logger_dir (`str`, *optional*, defaults to ""):
            Argument config_logger_dir.
        error_injection_rate (`int`, *optional*, defaults to 0):
            Argument error_injection_rate.
        error_injection_type (`str`, *optional*, defaults to "transient_error"):
            Argument error_injection_type.
        rerun_mode (`str`, *optional*, defaults to "disabled"):
            Argument rerun_mode.
        enable_msc (`bool`, *optional*, defaults to `True`):
            Argument enable_msc.
        kitchen_config_file (`str`):
            Argument kitchen_config_file.
        kitchen_recipe_number (`int`):
            Argument kitchen_recipe_number.
        sft (`bool`, *optional*, defaults to `False`):
            Argument sft.
        sft_tokenizer_prompt_format (`str`, *optional*, defaults to "nemotron-h-aligned"):
            Argument sft_tokenizer_prompt_format.
        pad_token_id (`int`, *optional*, defaults to 0):
            Argument pad_token_id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Argument bos_token_id.
        eos_token_id (`int`, *optional*, defaults to 0):
            Argument eos_token_id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Argument tie_word_embeddings.
        use_cache (`bool`, *optional*, defaults to `True):
            Argument use_cache.
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
        mrope_section: int = None,
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
        rampup_batch_size: None = None,
        decrease_batch_size_if_needed: bool = False,
        recompute_activations: bool = False,
        recompute_granularity: str = None,
        check_for_nan_in_loss_and_grad: bool = True,
        check_for_spiky_loss: bool = False,
        check_for_large_grads: bool = False,
        distribute_saved_activations: bool = False,
        recompute_method: str = None,
        recompute_num_layers: int = None,
        recompute_modules: str = None,
        clone_scatter_output_in_embedding: bool = True,
        profile: bool = False,
        profile_step_start: int = 10,
        profile_step_end: int = 12,
        iterations_to_skip: int = [],
        result_rejected_tracker_filename: str = None,
        enable_gloo_process_groups: bool = True,
        use_pytorch_profiler: bool = False,
        profile_ranks: int = [0],
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
        high_priority_stream_groups: str = [],
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
        cp_comm_type: str = ["p2p"],
        hierarchical_context_parallel_sizes: int = None,
        nccl_communicator_config_path: str = None,
        use_tp_pp_dp_mapping: bool = False,
        replication: bool = False,
        replication_jump: int = None,
        replication_factor: int = 2,
        eval_iters: int = 100,
        eval_interval: int = 1000,
        test_mode: bool = False,
        skip_train: bool = False,
        data_path: None = None,
        split: str = None,
        train_data_path: None = None,
        valid_data_path: None = None,
        test_data_path: None = None,
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
        tiktoken_special_tokens: str = None,
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
        retriever_report_topk_accuracies: int = [],
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
        spec: str = None,
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
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
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
        self.rotary_seq_len_interpolation_factor: int = rotary_seq_len_interpolation_factor
        self.use_rope_scaling: bool = use_rope_scaling
        self.rope_scaling_factor: float = rope_scaling_factor
        self.no_rope_freq: Any = no_rope_freq
        self.add_position_embedding: bool = add_position_embedding
        self.mrope_section: int = mrope_section
        self.make_vocab_size_divisible_by: int = make_vocab_size_divisible_by
        self.normalization: None = normalization
        self.norm_epsilon: float = norm_epsilon
        self.apply_layernorm_1p: bool = apply_layernorm_1p
        self.apply_residual_connection_post_layernorm: bool = apply_residual_connection_post_layernorm
        self.openai_gelu: bool = openai_gelu
        self.squared_relu: bool = squared_relu
        self.swiglu: bool = swiglu
        self.onnx_safe: bool = onnx_safe
        self.bert_binary_head: bool = bert_binary_head
        self.untie_embeddings_and_output_weights: bool = untie_embeddings_and_output_weights
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
        self.rampup_batch_size: None = rampup_batch_size
        self.decrease_batch_size_if_needed: bool = decrease_batch_size_if_needed
        self.recompute_activations: bool = recompute_activations
        self.recompute_granularity: str = recompute_granularity
        self.check_for_nan_in_loss_and_grad: bool = check_for_nan_in_loss_and_grad
        self.check_for_spiky_loss: bool = check_for_spiky_loss
        self.check_for_large_grads: bool = check_for_large_grads
        self.distribute_saved_activations: bool = distribute_saved_activations
        self.recompute_method: str = recompute_method
        self.recompute_num_layers: int = recompute_num_layers
        self.recompute_modules: str = recompute_modules
        self.clone_scatter_output_in_embedding: bool = clone_scatter_output_in_embedding
        self.profile: bool = profile
        self.profile_step_start: int = profile_step_start
        self.profile_step_end: int = profile_step_end
        self.iterations_to_skip: int = iterations_to_skip
        self.result_rejected_tracker_filename: str = result_rejected_tracker_filename
        self.enable_gloo_process_groups: bool = enable_gloo_process_groups
        self.use_pytorch_profiler: bool = use_pytorch_profiler
        self.profile_ranks: int = profile_ranks
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
        self.check_weight_hash_across_dp_replicas_interval: int = check_weight_hash_across_dp_replicas_interval
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
        self.use_torch_optimizer_for_cpu_offload: bool = use_torch_optimizer_for_cpu_offload
        self.overlap_cpu_optimizer_d2h_h2d: bool = overlap_cpu_optimizer_d2h_h2d
        self.pin_cpu_grads: bool = pin_cpu_grads
        self.pin_cpu_params: bool = pin_cpu_params
        self.dataloader_type: str = dataloader_type
        self.async_tensor_model_parallel_allreduce: bool = async_tensor_model_parallel_allreduce
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
        self.pipeline_model_parallel_comm_backend: str = pipeline_model_parallel_comm_backend
        self.high_priority_stream_groups: str = high_priority_stream_groups
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
        self.use_checkpoint_opt_param_scheduler: bool = use_checkpoint_opt_param_scheduler
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
        self.use_tokenizer_model_from_checkpoint_args: bool = use_tokenizer_model_from_checkpoint_args
        self.exit_on_missing_checkpoint: bool = exit_on_missing_checkpoint
        self.use_dist_ckpt_deprecated: bool = use_dist_ckpt_deprecated
        self.use_persistent_ckpt_worker: bool = use_persistent_ckpt_worker
        self.auto_detect_ckpt_format: bool = auto_detect_ckpt_format
        self.dist_ckpt_format_deprecated: None = dist_ckpt_format_deprecated
        self.ckpt_format: None = ckpt_format
        self.ckpt_convert_format: None = ckpt_convert_format
        self.ckpt_convert_save: None = ckpt_convert_save
        self.ckpt_convert_update_legacy_dist_opt_format: bool = ckpt_convert_update_legacy_dist_opt_format
        self.ckpt_fully_parallel_save_deprecated: bool = ckpt_fully_parallel_save_deprecated
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
        self.accumulate_allreduce_grads_in_fp32: bool = accumulate_allreduce_grads_in_fp32
        self.fp16_lm_cross_entropy: bool = fp16_lm_cross_entropy
        self.disable_bf16_reduced_precision_matmul: bool = disable_bf16_reduced_precision_matmul
        self.reuse_grad_buf_for_mxfp8_param_ag: bool = reuse_grad_buf_for_mxfp8_param_ag
        self.tensor_model_parallel_size: int = tensor_model_parallel_size
        self.encoder_tensor_model_parallel_size: int = encoder_tensor_model_parallel_size
        self.pipeline_model_parallel_size: int = pipeline_model_parallel_size
        self.encoder_pipeline_model_parallel_size: int = encoder_pipeline_model_parallel_size
        self.pipeline_model_parallel_split_rank: int = pipeline_model_parallel_split_rank
        self.decoder_first_pipeline_num_layers: int = decoder_first_pipeline_num_layers
        self.decoder_last_pipeline_num_layers: int = decoder_last_pipeline_num_layers
        self.pipeline_model_parallel_layout: str = pipeline_model_parallel_layout
        self.model_parallel_size: int = model_parallel_size
        self.num_layers_per_virtual_pipeline_stage: int = num_layers_per_virtual_pipeline_stage
        self.num_virtual_stages_per_pipeline_rank: int = num_virtual_stages_per_pipeline_rank
        self.microbatch_group_size_per_vp_stage: int = microbatch_group_size_per_vp_stage
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
        self.ddp_pad_buckets_for_high_nccl_busbw: bool = ddp_pad_buckets_for_high_nccl_busbw
        self.ddp_average_in_collective: bool = ddp_average_in_collective
        self.overlap_param_gather: bool = overlap_param_gather
        self.overlap_param_gather_with_optimizer_step: bool = overlap_param_gather_with_optimizer_step
        self.align_param_gather: bool = align_param_gather
        self.scatter_gather_tensors_in_pipeline: bool = scatter_gather_tensors_in_pipeline
        self.use_ring_exchange_p2p: bool = use_ring_exchange_p2p
        self.local_rank: int = local_rank
        self.lazy_mpu_init: bool = lazy_mpu_init
        self.account_for_embedding_in_pipeline_split: bool = account_for_embedding_in_pipeline_split
        self.account_for_loss_in_pipeline_split: bool = account_for_loss_in_pipeline_split
        self.use_distributed_optimizer: bool = use_distributed_optimizer
        self.nccl_ub: bool = nccl_ub
        self.use_sharp: bool = use_sharp
        self.use_custom_fsdp: bool = use_custom_fsdp
        self.init_model_with_meta_device: bool = init_model_with_meta_device
        self.data_parallel_sharding_strategy: str = data_parallel_sharding_strategy
        self.gradient_reduce_div_fusion: bool = gradient_reduce_div_fusion
        self.fsdp_double_buffer: bool = fsdp_double_buffer
        self.suggested_communication_unit_size: int = suggested_communication_unit_size
        self.keep_fp8_transpose_cache_when_using_custom_fsdp: bool = keep_fp8_transpose_cache_when_using_custom_fsdp
        self.num_distributed_optimizer_instances: int = num_distributed_optimizer_instances
        self.use_torch_fsdp2: bool = use_torch_fsdp2
        self.torch_fsdp2_reshard_after_forward: bool = torch_fsdp2_reshard_after_forward
        self.context_parallel_size: int = context_parallel_size
        self.cp_comm_type: str = cp_comm_type
        self.hierarchical_context_parallel_sizes: int = hierarchical_context_parallel_sizes
        self.nccl_communicator_config_path: str = nccl_communicator_config_path
        self.use_tp_pp_dp_mapping: bool = use_tp_pp_dp_mapping
        self.replication: bool = replication
        self.replication_jump: int = replication_jump
        self.replication_factor: int = replication_factor
        self.eval_iters: int = eval_iters
        self.eval_interval: int = eval_interval
        self.test_mode: bool = test_mode
        self.skip_train: bool = skip_train
        self.data_path: None = data_path
        self.split: str = split
        self.train_data_path: None = train_data_path
        self.valid_data_path: None = valid_data_path
        self.test_data_path: None = test_data_path
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
        self.create_attention_mask_in_dataloader: bool = create_attention_mask_in_dataloader
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
        self.tiktoken_special_tokens: str = tiktoken_special_tokens
        self.adlr_autoresume: bool = adlr_autoresume
        self.adlr_autoresume_interval: int = adlr_autoresume_interval
        self.ict_head_size: int = ict_head_size
        self.biencoder_projection_dim: int = biencoder_projection_dim
        self.biencoder_shared_query_context_model: bool = biencoder_shared_query_context_model
        self.ict_load: str = ict_load
        self.bert_load: str = bert_load
        self.titles_data_path: str = titles_data_path
        self.query_in_block_prob: float = query_in_block_prob
        self.use_one_sent_docs: bool = use_one_sent_docs
        self.evidence_data_path: str = evidence_data_path
        self.retriever_report_topk_accuracies: int = retriever_report_topk_accuracies
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
        self.moe_shared_expert_intermediate_size: int = moe_shared_expert_intermediate_size
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
        self.heterogeneous_layers_config_encoded_json: str = heterogeneous_layers_config_encoded_json
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
        self.inference_batch_times_seqlen_threshold: int = inference_batch_times_seqlen_threshold
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
        self.inference_dynamic_batching_buffer_size_gb: float = inference_dynamic_batching_buffer_size_gb
        self.inference_dynamic_batching_chunk_size: int = inference_dynamic_batching_chunk_size
        self.inference_dynamic_batching_buffer_guaranteed_fraction: float = (
            inference_dynamic_batching_buffer_guaranteed_fraction
        )
        self.inference_dynamic_batching_buffer_overflow_factor: float = (
            inference_dynamic_batching_buffer_overflow_factor
        )
        self.inference_dynamic_batching_max_requests_override: int = inference_dynamic_batching_max_requests_override
        self.inference_dynamic_batching_max_tokens_override: int = inference_dynamic_batching_max_tokens_override
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
        self.spec: str = spec
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
        self.inprocess_monitor_thread_interval: float = inprocess_monitor_thread_interval
        self.inprocess_monitor_process_interval: float = inprocess_monitor_process_interval
        self.inprocess_progress_watchdog_interval: float = inprocess_progress_watchdog_interval
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
        self.tie_word_embeddings: bool = tie_word_embeddings
        self.use_cache: bool = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MegatronConfig"]
