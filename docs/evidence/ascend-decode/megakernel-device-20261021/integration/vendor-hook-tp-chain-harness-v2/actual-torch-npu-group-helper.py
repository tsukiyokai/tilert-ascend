def _patched_new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """

    if group_name in dist_c10d._world.pg_names.values():
        raise ValueError(
            "The specified group name has already been "
            "created, please use a different group name"
        )

    if device_id is not None and (device_id.index is None or device_id.type == "cpu"):
        raise ValueError(
            "init_process_group device_id parameter must be an accelerator with an index"
        )

    # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
    _check_valid_timeout(timeout)

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = dist_c10d._world.pg_map[existing_group]
            return existing_group, prefix_store

    group_desc = "undefined" if group_desc is None else group_desc

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = len(global_ranks_in_group) == 0

    # nccl and potentially other backends allow creation of
    # communicators based on pre-existing ones, which can save
    # initialization time.  Due to lazy initialization of
    # communicators in some backends, we have to be careful and only
    # split when we *know* the default PG has already started communicator initialization.
    # We know this if we have bound a device id to the default pg (eager initialized).
    if is_initialized() and _get_default_group().bound_device_id:
        split_from = _get_split_source(_get_default_group())
    else:
        split_from = None

    # If this is a subgroup (which means group_ranks is specified),
    # we check if the current process is a member of the new group.
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            # If we are using `ncclCommSplit` (or similar split from
            # other APIs) to create the communicator, we will need to
            # call `ncclCommSplit` on *all* ranks in this new group's
            # parent group, even those not in the new group.  This is
            # a requirement of the NCCL API as otherwise we would get
            # out of sync.
            if split_from:
                split_from.perform_nocolor_split(_get_default_group().bound_device_id)
            return GroupMember.NON_GROUP_MEMBER, None

    prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    backend_config = BackendConfig(backend)
    # Set the default backend when single backend is passed in.
    if "," not in str(backend) and ":" not in str(backend):
        if backend not in Backend.backend_type_map:
            raise AssertionError(f"Unknown backend type {backend}")
        if backend == Backend.UNDEFINED:
            # Currently when backend is UNDEFINED, only one backend will be initialized
            # we use nccl (if cuda is available) or gloo as default backend
            # so we can correctly call getDefaultBackend which in ProcessGroup.
            if Backend.NCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.NCCL)
            elif Backend.HCCL in backend_config.get_device_backend_map().values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
            else:
                pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        else:
            pg._set_default_backend(Backend.backend_type_map[backend])
    # In order to correctly call pg._has_hooks(), we should set the default backend
    # when multi backend is passed in
    else:
        if Backend.NCCL in backend_config.device_backend_map.values():
            pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        elif Backend._plugins.keys():
            custom_backend = next(iter(Backend._plugins.keys()))
            if custom_backend in backend_config.device_backend_map.values():
                pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        else:
            pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    if device_id:
        pg.bound_device_id = device_id
    backend_class: torch._C._distributed_c10d.Backend
    for device, backend_str in backend_config.get_device_backend_map().items():
        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError(
                    "Distributed package doesn't have MPI built in."
                    " MPI is only included if you build PyTorch from"
                    " source on a host that has MPI installed."
                )
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER, None
            # create new process group with accurate rank and size
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(
                    backend_prefix_store,
                    backend_class.rank(),
                    backend_class.size(),
                )
                pg._set_default_backend(backend_type)
        elif backend_str == Backend.GLOO:
            # TODO: remove this check after lazy initialization is supported
            # if pg_options is not None:
            #     raise RuntimeError("GLOO options not supported")
            if not is_gloo_available():
                raise RuntimeError("Distributed package doesn't have Gloo built in")
            backend_class = ProcessGroupGloo(
                backend_prefix_store,
                group_rank,
                group_size,
                # pyrefly: ignore [bad-argument-type]
                timeout=timeout,
            )
            backend_class.options.global_ranks_in_group = global_ranks_in_group
            backend_class.options.group_name = group_name
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if backend_options is not None:
                if not isinstance(backend_options, ProcessGroupNCCL.Options):
                    raise AssertionError(
                        "Expected backend_options argument to be of type ProcessGroupNCCL.Options"
                    )
                if backend_options._timeout != timeout:
                    warnings.warn(
                        "backend_options._timeout was specified, "
                        "but timeout kwarg has a default value that will always override it. ",
                        stacklevel=2,
                    )
            else:
                # default backend_options for NCCL
                backend_options = ProcessGroupNCCL.Options()
                backend_options.is_high_priority_stream = False
            # pyrefly: ignore [bad-argument-type]
            backend_options._timeout = timeout

            if split_from:
                backend_options.split_from = split_from
                backend_options.split_color = _process_group_color(
                    global_ranks_in_group
                )
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            backend_class = ProcessGroupNCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            # TODO: once UCC plugin is fully deprecated, remove
            # is_ucc_available() from above elif-condition and raise
            # RuntimeError if is_ucc_available() returns false.

            backend_class = ProcessGroupUCC(
                backend_prefix_store,
                group_rank,
                group_size,
                # pyrefly: ignore [bad-argument-type]
                timeout=timeout,
            )
            backend_type = ProcessGroup.BackendType.UCC
        elif backend_str == Backend.XCCL:
            if not is_xccl_available():
                raise RuntimeError("Distributed package doesn't have XCCL built in")
            backend_options = ProcessGroupXCCL.Options()
            backend_options.global_ranks_in_group = global_ranks_in_group
            backend_options.group_name = group_name
            # pyrefly: ignore [bad-argument-type]
            backend_options._timeout = timeout
            backend_class = ProcessGroupXCCL(
                backend_prefix_store, group_rank, group_size, backend_options
            )
            backend_type = ProcessGroup.BackendType.XCCL
        else:
            if backend_str.upper() not in Backend._plugins:
                raise AssertionError(f"Unknown c10d backend type {backend_str.upper()}")

            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM

            if not extended_api:
                backend_class = creator_fn(
                    backend_prefix_store, group_rank, group_size, timeout
                )
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                # pyrefly: ignore [bad-argument-type]
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                backend_class = creator_fn(dist_backend_opts, backend_options)

        # Set sequence numbers for gloo and nccl backends.
        if backend_str == Backend.GLOO:
            if not isinstance(backend_class, ProcessGroupGloo):
                raise AssertionError(
                    f"Expected ProcessGroupGloo, got {type(backend_class)}"
                )
            backend_class._set_sequence_number_for_group()
        elif backend_str == Backend.NCCL:
            if not isinstance(backend_class, ProcessGroupNCCL):
                raise AssertionError(
                    f"Expected ProcessGroupNCCL, got {type(backend_class)}"
                )
            backend_class._set_sequence_number_for_group()

        # If the type is a subclass of ProcessGroup then return this process group immediately
        # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
        # ProcessGroup instance
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class  # type: ignore[assignment]
            break

        # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
        if (
            backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
            or backend_str.upper() in Backend._plugins
        ):
            # In debug mode and if GLOO is available, wrap in a wrapper PG that
            # enables enhanced collective checking for debuggability.
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info(
                        """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                GLOO is not available. Build with Gloo to
                                create a wrapper process group in debug mode
                                to aid collective desynchronization debugging."""
                    )
                else:
                    backend_class = _create_process_group_wrapper(
                        wrapped_pg=backend_class,
                        store_prefix=group_name,
                        store=backend_prefix_store,
                        rank=group_rank,
                        world_size=group_size,
                        # pyrefly: ignore [bad-argument-type]
                        timeout=timeout,
                    )

        # register only a single backend when all get_device_backend_map values are the same
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map():
                pg._register_backend(torch.device(device), backend_type, backend_class)

            # break out of outer loop to not create any more backends
            break

        pg._register_backend(torch.device(device), backend_type, backend_class)

    # set group_name and group_dsec to backend
    if group_name is None:
        raise AssertionError("group_name must not be None")
    if group_desc is None:
        raise AssertionError("group_desc must not be None")
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    if device_id and pg._get_backend(device_id).supports_splitting:
        eager_backend = pg._get_backend(device_id)
        eager_backend.eager_connect_single_device(device_id)

    # update global state
    dist_c10d._world.pg_map[pg] = (backend, prefix_store)
    dist_c10d._world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)

    dist_c10d._world.pg_backend_config[pg] = str(backend_config)
    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        dist_c10d._world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    dist_c10d._world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    dist_c10d._world.pg_to_tag[pg] = pg_tag
    return pg, prefix_store
